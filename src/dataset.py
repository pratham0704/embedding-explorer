import json
import random
import os
import re


CATEGORIES = [
    "payment_failure",
    "fraud_alert",
    "kyc_verification",
    "account_access",
    "refund_request",
    "loan_application",
    "nach_mandate",
    "upi_transaction",
    "card_dispute",
    "compliance",
]


TEMPLATES = {
    "payment_failure": [
        "Payment failed with {error_code}. Please retry.",
        "Transaction {txn_id} was declined by the issuing bank.",
        "Customer reported that UPI payment {txn_id} failed but the amount was debited.",
        "Payment attempt for CASE {case_id} was not completed due to gateway timeout.",
        "Card payment was rejected with {error_code} during checkout.",
        "NEFT transfer linked to {txn_id} failed and beneficiary did not receive funds.",
        "Payment was not approved because of insufficient balance.",
        "IMPS transaction {txn_id} failed, but customer received no reversal confirmation.",
        "Customer says the payment page showed success, but backend status is failed.",
        "Auto-debit payment failed for loan account due to NACH mandate issue.",
    ],

    "fraud_alert": [
        "Transaction {txn_id} was flagged for suspicious activity.",
        "Fraud alert generated for CASE {case_id} due to unusual spending behavior.",
        "Customer denies making transaction {txn_id} on their debit card.",
        "Multiple failed payment attempts detected from the same device fingerprint.",
        "Risk engine marked the UPI transaction as high risk.",
        "Account showed abnormal login and payment activity within a short time window.",
        "Transaction was not approved because fraud rules were triggered.",
        "Potential mule account behavior detected for CASE {case_id}.",
        "Customer reported unauthorized IMPS transfer from their savings account.",
        "Fraud monitoring system raised alert after velocity check failure.",
    ],

    "kyc_verification": [
        "KYC verification failed due to missing PAN document.",
        "Customer identity verification is not complete for CASE {case_id}.",
        "Aadhaar verification could not be completed because OTP was not received.",
        "KYC status is pending even after customer uploaded required documents.",
        "Customer submitted incorrect address proof during KYC verification.",
        "Video KYC was rejected because the face match score was below threshold.",
        "KYC documents for CASE {case_id} require manual review.",
        "Customer says CKYC record exists, but system is asking for fresh KYC.",
        "Identity verification failed due to mismatch in PAN and CIBIL details.",
        "KYC is complete, but account activation is still blocked.",
    ],

    "account_access": [
        "Customer is unable to log in after multiple OTP attempts.",
        "Account access was blocked after suspicious login activity.",
        "User says password reset link is not working.",
        "Customer cannot access account because mobile number is not updated.",
        "Login request was denied due to device change verification failure.",
        "Account is inactive and customer wants reactivation.",
        "Customer reports that biometric login is not enabled on the new device.",
        "Access to account was not restored after successful KYC verification.",
        "User is locked out after entering incorrect PIN several times.",
        "Customer can log in, but cannot view linked NBFC loan details.",
    ],

    "refund_request": [
        "Customer requested refund for failed transaction {txn_id}.",
        "Refund is not credited even though payment failed with {error_code}.",
        "Customer says amount was debited but merchant did not receive payment.",
        "Refund for UPI transaction {txn_id} is pending beyond TAT.",
        "Refund request linked to CASE {case_id} was rejected by operations team.",
        "Customer wants reversal for duplicate card charge.",
        "Payment failed, but automatic refund was not initiated.",
        "Customer claims refund was approved but not received in bank account.",
        "NEFT refund could not be processed due to invalid beneficiary details.",
        "Refund status shows completed, but customer denies receiving funds.",
    ],

    "loan_application": [
        "Loan application for CASE {case_id} is pending credit review.",
        "Customer is not eligible for the loan due to low CIBIL score.",
        "Loan request was rejected because income documents were incomplete.",
        "NBFC partner denied the application after bureau verification.",
        "Customer says CIBIL score is updated, but loan application still shows rejected.",
        "Application was not approved due to high existing EMI obligations.",
        "Customer wants to know why personal loan was declined.",
        "Loan approval is blocked because KYC verification is not complete.",
        "NPA history was detected during credit underwriting.",
        "Loan application was approved, but disbursal has not happened.",
    ],

    "nach_mandate": [
        "NACH mandate registration failed for customer loan account.",
        "Auto-debit was not processed because NACH mandate is inactive.",
        "Customer says NACH mandate was approved but EMI was not debited.",
        "Mandate setup for CASE {case_id} failed due to bank validation error.",
        "NACH debit was rejected by sponsor bank.",
        "Customer wants to cancel active NACH mandate.",
        "EMI payment failed because mandate approval is still pending.",
        "NACH mandate was denied due to name mismatch in bank records.",
        "Customer received debit notification but NACH transaction was not successful.",
        "Mandate status is active, but collection system shows no valid mandate.",
    ],

    "upi_transaction": [
        "UPI transaction {txn_id} is pending for more than 30 minutes.",
        "Customer entered wrong UPI ID and wants payment reversal.",
        "UPI collect request was not approved by the customer.",
        "Transaction {txn_id} failed due to PSP timeout.",
        "Customer says UPI payment succeeded, but merchant has no confirmation.",
        "UPI transaction was declined by NPCI switch.",
        "Customer reports duplicate debit for same UPI reference.",
        "UPI payment linked to CASE {case_id} is under dispute.",
        "Customer cannot complete UPI transaction because daily limit is exceeded.",
        "UPI refund is pending even after transaction failure.",
    ],

    "card_dispute": [
        "Customer raised dispute for card transaction {txn_id}.",
        "Card payment was not authorized by the customer.",
        "Chargeback request for CASE {case_id} is pending review.",
        "Customer says debit card was charged twice at merchant outlet.",
        "Card transaction was declined but amount was debited.",
        "Customer denies ATM withdrawal shown in statement.",
        "Dispute was rejected because merchant submitted valid proof.",
        "Customer wants provisional credit for unauthorized card transaction.",
        "Card network marked transaction as successful, but customer says it failed.",
        "Customer reports international card transaction without consent.",
    ],

    "compliance": [
        "CASE {case_id} requires compliance review due to AML risk.",
        "Customer account is not compliant with updated KYC norms.",
        "Suspicious transaction report needs review by compliance team.",
        "NBFC partner requested additional documents for regulatory verification.",
        "Account was frozen due to unresolved compliance alert.",
        "Customer is marked high risk after sanctions screening.",
        "Compliance approval is pending for transaction {txn_id}.",
        "Customer profile failed internal risk policy validation.",
        "Enhanced due diligence is required before account activation.",
        "NPA-linked borrower requires manual compliance clearance.",
    ],
}


ERROR_CODES = [
    "ERR_PAYMENT_4821",
    "ERR_GATEWAY_503",
    "ERR_INSUFFICIENT_FUNDS_3829",
    "ERR_CARD_DECLINED_441",
    "ERR_TIMEOUT_992",
]

TXN_IDS = [f"TXN-{random.randint(100000, 999999)}" for _ in range(100)]

CASE_IDS = [f"CASE-2024-{str(i).zfill(6)}" for i in range(1, 201)]

DOMAIN_TERMS = ["NACH", "NPA", "KYC", "CIBIL", "NBFC", "UPI", "NEFT", "IMPS"]


class Dataset:
    def generate(self, n=10000, save_path=None):
        tickets = []

        for ticket_id in range(1, n + 1):
            category = random.choice(CATEGORIES)
            template = random.choice(TEMPLATES[category])

            text = template.format(
            error_code=random.choice(ERROR_CODES),
            txn_id=random.choice(TXN_IDS),
            case_id=random.choice(CASE_IDS),
            ) + f" [Ref: {ticket_id}]"

            ticket = {
                "id": ticket_id,
                "category": category,
                "text": text,
                "metadata": {
                    "has_identifier": self._has_identifier(text),
                    "has_negation": self._has_negation(text),
                    "domain_terms": self._extract_domain_terms(text),
                },
            }

            tickets.append(ticket)

        if save_path:
            directory = os.path.dirname(save_path)

            if directory:
                os.makedirs(directory, exist_ok=True)

            with open(save_path, "w", encoding="utf-8") as file:
                json.dump(tickets, file, indent=2, ensure_ascii=False)

        return tickets

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")

        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _has_identifier(self, text):
        identifier_patterns = [
            r"\bERR_[A-Z_]+_\d+\b",
            r"\bTXN-\d+\b",
            r"\bCASE-2024-\d{6}\b",
        ]

        for pattern in identifier_patterns:
            if re.search(pattern, text):
                return True

        return False

    def _has_negation(self, text):
        negation_words = [
            "not",
            "no",
            "never",
            "rejected",
            "denied",
            "failed",
            "declined",
            "unable",
            "cannot",
            "blocked",
            "inactive",
            "incorrect", 
            "invalid", 
            "incomplete", 
            "mismatch"
        ]

        normalized_text = text.lower()

        for word in negation_words:
            pattern = rf"\b{re.escape(word)}\b"

            if re.search(pattern, normalized_text):
                return True

        return False

    def _extract_domain_terms(self, text):
        found_terms = []
        normalized_text = text.upper()

        for term in DOMAIN_TERMS:
            pattern = rf"\b{re.escape(term)}\b"

            if re.search(pattern, normalized_text):
                found_terms.append(term)

        return found_terms


if __name__ == "__main__":
    dataset = Dataset()

    tickets = dataset.generate(n=50, save_path="tickets.json")

    print(f"Generated {len(tickets)} tickets")
    print("\nSample tickets:")

    for ticket in tickets[:5]:
        print("-" * 100)
        print(json.dumps(ticket, indent=2))