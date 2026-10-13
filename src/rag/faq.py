"""Synthetic FAQ knowledge base mapped to Banking77 intents."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FAQ:
    intent: str
    question: str
    answer: str
    escalation: bool = False


FAQ_DB: list[FAQ] = [
    FAQ(
        "card_about_to_expire",
        "My card is about to expire. What should I do?",
        "We automatically mail a replacement card 2-4 weeks before expiration. If you haven't received it, you can order a new card in the app.",
    ),
    FAQ(
        "card_delivery_estimate",
        "When will my new card arrive?",
        "Standard delivery is 5-7 business days. Express delivery is 1-2 business days where available.",
    ),
    FAQ(
        "card_not_working",
        "My card is not working.",
        "Check that the card is activated and not frozen. If the problem persists, contact support.",
    ),
    FAQ(
        "direct_debit_payment_not_recognised",
        "I don't recognize a direct debit payment.",
        "You can dispute unrecognized direct debits in the app within 8 weeks of the debit date.",
        escalation=True,
    ),
    FAQ(
        "cancel_transfer",
        "Can I cancel a transfer?",
        "Transfers sent to another account in your name can sometimes be reversed. Transfers to third parties usually cannot be cancelled once processed.",
        escalation=True,
    ),
    FAQ(
        "exchange_rate",
        "What exchange rate will I get?",
        "We use the Mastercard/Visa exchange rate with no additional markup for most currencies.",
    ),
    FAQ(
        "top_up_by_bank_transfer_charge",
        "Is there a fee for topping up by bank transfer?",
        "Bank transfer top-ups are free. Instant card top-ups may carry a small fee depending on your plan.",
    ),
    FAQ(
        "balance_not_updated_after_bank_transfer",
        "My balance hasn't updated after a bank transfer.",
        "Bank transfers can take up to 2 business days. If it has been longer, please provide the transfer reference.",
        escalation=True,
    ),
    FAQ(
        "change_pin",
        "How do I change my PIN?",
        "You can change your PIN at any ATM that supports PIN services, or in the app under Card Settings.",
    ),
    FAQ(
        "compromised_card",
        "I think my card details have been stolen.",
        "Freeze your card immediately in the app and order a replacement. We also recommend reporting this to support.",
        escalation=True,
    ),
    FAQ(
        "age_limit",
        "Is there an age limit to open an account?",
        "You must be at least 18 years old to open a personal account.",
    ),
    FAQ(
        "verify_my_identity",
        "How do I verify my identity?",
        "Go to Profile > Verification and follow the prompts. You'll need a valid ID and a selfie.",
    ),
    FAQ(
        "visa_or_mastercard",
        "Do you offer Visa or Mastercard?",
        "We issue both Visa and Mastercard depending on your region and product.",
    ),
    FAQ(
        "transaction_charged_twice",
        "I was charged twice for the same transaction.",
        "Duplicate authorizations usually drop off within 7 business days. If the charge settles twice, you can dispute it.",
        escalation=True,
    ),
    FAQ(
        "wrong_amount_of_cash_received",
        "I received the wrong amount of cash from an ATM.",
        "Please keep the receipt and contact support with the ATM location, date, and amount.",
        escalation=True,
    ),
]


def faq_documents() -> list[dict]:
    return [
        {
            "page_content": f"Intent: {f.intent}\nQ: {f.question}\nA: {f.answer}",
            "metadata": {
                "intent": f.intent,
                "escalation": f.escalation,
                "question": f.question,
                "answer": f.answer,
            },
        }
        for f in FAQ_DB
    ]
