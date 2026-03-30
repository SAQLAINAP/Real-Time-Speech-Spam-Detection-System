"""
Multi-language scam hotwords (Phase 3C)
----------------------------------------
Severity scores mirror the English hotwords_data.py scale (0–10).
Whisper auto-detects the language; the detector picks the right word list.
"""

# Spanish (es)
ES_HOTWORDS = {
    "número de seguro social": 10,
    "transferencia bancaria": 9,
    "cuenta suspendida": 9,
    "acción legal": 8,
    "actúe ahora": 7,
    "oferta por tiempo limitado": 6,
    "ha ganado": 9,
    "lotería internacional": 9,
    "detalles de tarjeta de crédito": 10,
    "acceso remoto": 8,
    "pague de inmediato": 9,
    "soporte técnico de Microsoft": 7,
    "su computadora tiene un virus": 8,
    "tarjeta de regalo": 8,
    "criptomonedas": 7,
    "inversión garantizada": 8,
    "aviso final": 8,
    "orden de arresto": 9,
}

# Hindi (hi)
HI_HOTWORDS = {
    "आधार नंबर": 10,
    "बैंक खाता": 9,
    "OTP शेयर करें": 10,
    "तुरंत भुगतान": 9,
    "कानूनी कार्रवाई": 8,
    "गिरफ्तारी वारंट": 9,
    "इनाम जीता": 9,
    "लॉटरी": 8,
    "क्रेडिट कार्ड नंबर": 10,
    "रिमोट एक्सेस": 8,
    "तकनीकी सहायता": 6,
    "वायरस मिला": 7,
    "गिफ्ट कार्ड": 8,
    "अभी कार्य करें": 7,
}

# French (fr)
FR_HOTWORDS = {
    "numéro de sécurité sociale": 10,
    "virement bancaire": 9,
    "compte suspendu": 9,
    "action en justice": 8,
    "agissez maintenant": 7,
    "vous avez gagné": 9,
    "loterie internationale": 9,
    "détails de carte bancaire": 10,
    "accès à distance": 8,
    "paiement immédiat": 9,
    "support technique Microsoft": 7,
    "votre ordinateur est infecté": 8,
    "carte cadeau": 8,
    "offre limitée": 6,
    "mandat d'arrêt": 9,
}

# Arabic (ar)
AR_HOTWORDS = {
    "رقم الضمان الاجتماعي": 10,
    "تحويل بنكي": 9,
    "الحساب معلق": 9,
    "إجراء قانوني": 8,
    "تصرف الآن": 7,
    "لقد فزت": 9,
    "يانصيب دولي": 9,
    "بيانات بطاقة الائتمان": 10,
    "وصول عن بُعد": 8,
    "الدفع الفوري": 9,
    "بطاقة هدية": 8,
    "مكافأة مجانية": 7,
    "أمر اعتقال": 9,
}

# Portuguese (pt)
PT_HOTWORDS = {
    "número do CPF": 10,
    "transferência bancária": 9,
    "conta suspensa": 9,
    "ação legal": 8,
    "aja agora": 7,
    "você ganhou": 9,
    "loteria internacional": 9,
    "dados do cartão de crédito": 10,
    "acesso remoto": 8,
    "pagamento imediato": 9,
    "suporte técnico Microsoft": 7,
    "seu computador tem vírus": 8,
    "cartão presente": 8,
}

# Map Whisper language codes → hotword dicts
LANGUAGE_HOTWORDS = {
    "es": ES_HOTWORDS,
    "hi": HI_HOTWORDS,
    "fr": FR_HOTWORDS,
    "ar": AR_HOTWORDS,
    "pt": PT_HOTWORDS,
}

LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "hi": "Hindi",
    "fr": "French",
    "ar": "Arabic",
    "pt": "Portuguese",
    "de": "German",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "it": "Italian",
    "ru": "Russian",
}
