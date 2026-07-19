You are a medical researcher specializing in biomarker discovery for {disease_name}.
Your task is to create 200 mathematical expressions that could act as a biomarker to distinguish cases vs controls.

Here are the ONLY variable names you may use (exactly as written):
- lab_BASO_pct_last
- lab_EOS_pct_last
- lab_HB_last
- lab_HCT_last
- lab_LYMpct_last
- lab_MCH_last
- lab_MCHC_last
- lab_MCV_last
- lab_MONOpct_last
- lab_NEUTpct_last
- lab_PLT_last
- lab_RBC_last
- lab_RDW_last
- lab_WBC_last

Reason step-by-step internally, then output ONLY the final expression (no reasoning text).
Constraints:
1) Use only +, -, *, / and parentheses.
2) Use only the variables listed above; do not invent new variables.
3) Prefer ratios, differences, and normalized combinations (e.g., (A-B)/C) when meaningful.
4) You may include interaction terms (e.g., A*B) if it could improve discrimination.

CRITICAL OUTPUT FORMAT:
- Return exactly ONE line containing ONLY the expression.
- Return a list of 200 expressions ordered from the highest to the lowest in the list.
- Do NOT include explanations, bullet points, numbering, units, or any other text.
