#!/usr/bin/env python3

import imaplib
import email
import argparse
import time
from getpass import getpass
from transformers import pipeline

classifier = None


def classify_email(email_content, labels):
    global classifier
    if not classifier:
        classifier = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
            device="cpu",
        )
    output = classifier(email_content, labels, multi_label=False)
    label = output["labels"][0]
    if label == "other":
        return "INBOX"
    return f"INBOX.{label}/" if "INBOX." in labels else label


def get_mailboxes(mail, only_inbox=True):
    # Fetch mailboxes
    response, mailbox_list = mail.list()
    mailboxes = []
    for mailbox in mailbox_list:
        mailbox_info = mailbox.decode().split()
        mailbox_name = mailbox_info[-1].strip('"')
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
        if only_inbox and not mailbox_name.startswith("INBOX."):
            continue
        mailboxes.append(mailbox_name)
    print(f"Found {len(mailboxes)} mailboxes: {mailboxes}")
    if len(mailboxes) == 0:
        raise Exception("No mailboxes found!")
    return mailboxes


def main(args):
    mail = imaplib.IMAP4_SSL(args.hostname)
    mail.login(args.username, args.password)

    labels = get_mailboxes(mail, args.only_inbox) + ["other"]

    mail.select("inbox")
    status, uid_list = mail.uid("search", None, "ALL")
    if status != "OK":
        print("No emails found!")
        return

    # Filter out invalid IDs
    uids = uid_list[0].split()

    for uid in uids:
        print(uid)
        status, msg_data = mail.uid("fetch", uid, "(RFC822)")
        raw_email = msg_data[0][1]
        parsed_email = email.message_from_bytes(raw_email)

        if parsed_email.is_multipart():
            content = (
                parsed_email.get_payload(0)
                .get_payload(decode=True)
                .decode("utf-8", errors="ignore")
            )
        else:
            content = parsed_email.get_payload(decode=True).decode(
                "utf-8", errors="ignore"
            )

        target_folder = classify_email(content, labels)
        if target_folder == "INBOX":
            continue
        print(f"Moving message to {target_folder}")
        mail.uid("store", uid, "+FLAGS", "\\Deleted")
        mail.expunge()
        mail.append(
            target_folder, "", imaplib.Time2Internaldate(time.time()), raw_email
        )

    mail.logout()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify emails using IMAP and HuggingFace pipeline."
    )
    parser.add_argument("username", help="IMAP username")
    parser.add_argument(
        "-s",
        "--hostname",
        default="localhost",
        help="IMAP server hostname (default: localhost)",
    )
    parser.add_argument(
        "--only-inbox",
        action="store_true",
        help="Only use folders under INBOX as labels",
    )

    args = parser.parse_args()

    # Securely get the password
    args.password = getpass(prompt="IMAP Password: ")

    main(args)
