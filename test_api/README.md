# API smoke tests

Run from the **FrankenFit repo root** so imports and `.env` resolve consistently.

```bash
cd /Users/elekal/PyCharmProjects/FrankenFit
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r test_api/requirements.txt
cp test_api/.env.example .env
# Edit .env with your keys

python test_api/test_gemini.py
python test_api/test_tavily.py
```

Partner smokes (auto-skip with exit `0` when the key is missing):

```bash
python test_api/test_pioneer.py   # lists /base-models, confirms GLiNER 2 is visible
python test_api/test_entire.py    # placeholder until Entire docs land
```

Exit code `0` means success, skipped (missing key), or quota/rate-limit (Gemini may print a WARN). Non-zero means an unexpected failure.

**Note:** `google-generativeai` shows a deprecation warning; migrate to `google-genai` when convenient ([migration](https://github.com/google-gemini/deprecated-generative-ai-python)).
