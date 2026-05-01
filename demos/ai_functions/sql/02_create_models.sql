-- ----------------------------------------------------------------------------
-- AI Functions demo — register the seven AI Function models
--
-- Each AI Function is a self-contained IRISModel file in
--   /opt/irisapp/demos/ai_functions/iris_models/
-- The CREATE MODEL syntax uses IntegratedML's Custom Models USING clause:
--   - "pathtoclassifiers" / "pathtoregressors" → directory IRIS scans
--   - "iscmodelsdisabled": 1 → skip ISC built-ins, use only our IRISModels
--   - "userparams" → forwarded to IRISModel.__init__ as **kwargs
--
-- NOTE: Filenames (without .py) become the model identifier inside the
-- pathtoclassifiers directory. Each CREATE MODEL below will pick up exactly
-- one IRISModel file because the demo deploys the AI Function it needs into
-- a per-function staging directory at training time. See
-- scripts/deploy_models.py for the staging logic.
-- ----------------------------------------------------------------------------

-- ----- AI_SENTIMENT -----
DROP MODEL IF EXISTS AISentiment;
CREATE MODEL AISentiment
PREDICTING (sentiment)
FROM AIFunctions.ProductReviews
USING {
    "pathtoclassifiers": "/opt/irisapp/demos/ai_functions/iris_models/_staging/ai_sentiment",
    "iscmodelsdisabled": 1
};
TRAIN MODEL AISentiment;

-- ----- AI_CLASSIFY -----
DROP MODEL IF EXISTS AIClassify;
CREATE MODEL AIClassify
PREDICTING (department)
FROM AIFunctions.SupportTickets
USING {
    "pathtoclassifiers": "/opt/irisapp/demos/ai_functions/iris_models/_staging/ai_classify",
    "iscmodelsdisabled": 1,
    "userparams": {
        "categories": ["billing", "technical", "sales", "returns"]
    }
};
TRAIN MODEL AIClassify;

-- ----- AI_SUMMARIZE (executive summaries) -----
DROP MODEL IF EXISTS AISummarize;
CREATE MODEL AISummarize
PREDICTING (executive_summary)
FROM AIFunctions.MarketResearch
USING {
    "pathtoclassifiers": "/opt/irisapp/demos/ai_functions/iris_models/_staging/ai_summarize",
    "iscmodelsdisabled": 1,
    "userparams": {"max_sentences": 3}
};
TRAIN MODEL AISummarize;

-- ----- AI_TRANSLATE (Spanish/French/German/Portuguese -> English) -----
DROP MODEL IF EXISTS AITranslate;
CREATE MODEL AITranslate
PREDICTING (translated_comment)
FROM AIFunctions.UserComments
USING {
    "pathtoclassifiers": "/opt/irisapp/demos/ai_functions/iris_models/_staging/ai_translate",
    "iscmodelsdisabled": 1,
    "userparams": {"source_lang": "Spanish", "target_lang": "English"}
};
TRAIN MODEL AITranslate;

-- ----- EMBED_TEXT (128-dim hashing embedding) -----
DROP MODEL IF EXISTS EmbedText;
CREATE MODEL EmbedText
PREDICTING (content_vector)
FROM AIFunctions.Articles
USING {
    "pathtoclassifiers": "/opt/irisapp/demos/ai_functions/iris_models/_staging/embed_text",
    "iscmodelsdisabled": 1,
    "userparams": {"dim": 128}
};
TRAIN MODEL EmbedText;

-- ----- AI_EXTRACT (renewal date from contracts) -----
DROP MODEL IF EXISTS AIExtractRenewalDate;
CREATE MODEL AIExtractRenewalDate
PREDICTING (renewal_date)
FROM AIFunctions.LegalDocuments
USING {
    "pathtoclassifiers": "/opt/irisapp/demos/ai_functions/iris_models/_staging/ai_extract",
    "iscmodelsdisabled": 1,
    "userparams": {"question": "What is the contract renewal date?"}
};
TRAIN MODEL AIExtractRenewalDate;

-- ----- AI_COMPLETE (Claude-backed marketing copy generation) -----
DROP MODEL IF EXISTS AIComplete;
CREATE MODEL AIComplete
PREDICTING (product_copy)
FROM AIFunctions.ProductPrompts
USING {
    "pathtoclassifiers": "/opt/irisapp/demos/ai_functions/iris_models/_staging/ai_complete",
    "iscmodelsdisabled": 1,
    "userparams": {
        "model_id": "claude-haiku-4-5",
        "max_tokens": 256,
        "system_prompt": "You are a concise, brand-friendly product copywriter."
    }
};
TRAIN MODEL AIComplete;
