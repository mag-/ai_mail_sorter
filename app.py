#!/usr/bin/env python3

import imaplib
import email
import argparse
import os
import time
from getpass import getpass
from transformers import pipeline
import torch

classifier = None


def classify_email(email_content, labels, prefix=None):
    global classifier
    if not classifier:
        # Check if CUDA (NVIDIA GPU) is available
        if torch.cuda.is_available():
            device = "cuda"
        elif (
            torch.backends.mps.is_available()
        ):  # Check for Metal Performance Shaders (MPS) for Apple devices (available in newer torch versions)
            device = "mps"
        else:
            device = "cpu"
        classifier = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
            device=device,
        )
    if not prefix:
        prefix = ""
    modified_labels = [label[len(prefix) :] if label.startswith(prefix) else label for label in labels]
    output = classifier(email_content, modified_labels, multi_label=False)
    label = output["labels"][0]
    if label == "other":
        return "INBOX"
    if prefix + label in labels:
        return prefix + label
    return label


def extract_text_from_part(part):
    content_type = part.get_content_type()
    content_disposition = str(part.get("Content-Disposition"))

    # Check for attachments
    if "attachment" in content_disposition:
        return ""

    # Handle text types
    if content_type == "text/plain":
        return part.get_payload(decode=True).decode("utf-8", errors="ignore")
    elif content_type == "text/html":
        return part.get_payload(decode=True).decode("utf-8", errors="ignore")

    # Handle multipart types
    if part.is_multipart():
        text_content = ""
        for subpart in part.get_payload():
            text_content += extract_text_from_part(subpart)
        return text_content

    return ""


def move_email(mail, uid, source_folder, destination_folder, dry_run=False):
    if dry_run:
        print(f"[DRY RUN] Would move email with UID {uid} from {source_folder} to {destination_folder}")
        return

    # Select the source folder
    mail.select(source_folder)

    if " " in destination_folder:
        destination_folder = f'"{destination_folder}"'

    # Copy the email to the destination folder
    result, _ = mail.uid("COPY", uid, destination_folder)
    if result != "OK":
        raise Exception(f"Failed to copy email UID {uid} to {destination_folder}")

    # Mark the original email for deletion
    mail.uid("STORE", uid, "+FLAGS", "\\Deleted")

    # Expunge (permanently remove) emails marked for deletion
    mail.expunge()


def parse_mailbox_name(mailbox_info):
    # This function extracts the mailbox name from the mailbox info string.
    # It assumes the format: *(attributes) "delimiter" "mailbox_name"*
    # and extracts the mailbox_name which is the last quoted string.

    quote_indices = [pos for pos, char in enumerate(mailbox_info) if char == '"']
    if len(quote_indices) < 2:
        return None

    # Extract the mailbox name from the last pair of quotes
    start_index = quote_indices[-2] + 1
    end_index = quote_indices[-1]
    return mailbox_info[start_index:end_index]


def get_mailboxes(mail, prefix):
    # Fetch mailboxes
    response, mailbox_list = mail.list()
    mailboxes = []
    for mailbox in mailbox_list:
        mailbox_name = parse_mailbox_name(mailbox.decode())
        if not mailbox_name:
            continue

        # Filter criteria
        m = mailbox_name.lower()
        if (
            "archive" in m
            or "sent" in m
            or "drafts" in m
            or "trash" in m
            or "bin" in m
            or "deleted" in m
            or "flagged" in m
        ):
            continue
        if prefix and not mailbox_name.lower().startswith(prefix.lower()):
            continue
        mailboxes.append(mailbox_name)

    print(f"Found {len(mailboxes)} mailboxes: {mailboxes}")
    if len(mailboxes) == 0:
        raise Exception("No mailboxes found!")
    return mailboxes


def main(args):
    mail = imaplib.IMAP4_SSL(args.hostname)
    mail.login(args.username, args.password)

    labels = get_mailboxes(mail, args.prefix) + ["other"]

    mail.select("inbox")
    status, uid_list = mail.uid("search", None, "ALL")
    if status != "OK":
        print("No emails found!")
        return

    # Filter out invalid IDs
    uids = uid_list[0].split()

    for uid in uids:
        print(f"Processing message {uid}")
        status, msg_data = mail.uid("fetch", uid, "(RFC822)")
        raw_email = msg_data[0][1]
        parsed_email = email.message_from_bytes(raw_email)
        if args.dry_run:
            print(f"Subject: {parsed_email['subject']}")
        content = parsed_email["subject"]
        if parsed_email.is_multipart():
            content = extract_text_from_part(parsed_email)
        if content == "":
            content = parsed_email.get_payload(decode=True).decode("utf-8", errors="ignore")
        target_folder = classify_email(content, labels, args.prefix)
        if target_folder == "INBOX":
            continue
        print(f"Moving message to {target_folder}")
        move_email(mail, uid, "INBOX", target_folder, args.dry_run)

    mail.logout()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify emails using IMAP and HuggingFace pipeline.")
    parser.add_argument("-u", "--username", help="IMAP username")
    parser.add_argument(
        "-s",
        "--hostname",
        default="localhost",
        help="IMAP server hostname (default: localhost)",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Only use folders with the provided prefix as labels",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without actually moving emails.",
    )

    args = parser.parse_args()

    # Securely get the password
    if not args.username:
        args.username = os.environ.get("AUTO_EMAIL_CLASSIFIER_USERNAME")
        if not args.username:
            args.username = input("IMAP Username: ")
    args.password = os.environ.get("AUTO_EMAIL_CLASSIFIER_PASSWORD")
    if not args.password:
        args.password = getpass(prompt="IMAP Password: ")
    if not args.hostname:
        args.hostname = os.environ.get("AUTO_EMAIL_CLASSIFIER_HOSTNAME", "localhost")
    main(args)
