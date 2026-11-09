# Lab Answers: Student Performance Prediction System

## 1. SAD mini-analysis
| SAD element | Response |
|---|---|
| Problem statement | Academic risk is often noticed only after poor results. The prototype explores early identification of students who may need academic support, using available student indicators. |
| Users | Faculty member or academic analyst. |
| Inputs | Student attributes and academic indicators: study time, absences, past failures, G1, G2, and other columns from the dataset. |
| Process | Validate uploaded CSV columns → preprocess (impute, scale, one-hot encode) → apply the trained Random Forest → classify risk. |
| Outputs | AT RISK / NOT AT RISK label per record, a count of predicted at-risk records, a downloadable results CSV, and a model evaluation summary. |
| Key limitation | Trained on one Portuguese secondary-school dataset; it may not generalize to another institution or population. |

## 2. Model evaluation results (fill in from YOUR run of `python train_model.py`)
| Metric | Your value |
|---|---|
| Accuracy | ____ |
| Precision (At risk) | ____ |
| Recall (At risk) | ____ |
| F1-score (At risk) | ____ |

## 3. Testing checklist (tick after you run each one)
| Test | Expected result | Pass? |
|---|---|---|
| Run train_model.py | CSV and model files created without errors | ☐ |
| Run Streamlit app | Dashboard loads in browser | ☐ |
| Upload sample CSV | Predictions displayed | ☐ |
| Upload CSV missing a required column | Helpful error message shown (delete a column from the sample file to test) | ☐ |
| Download results | CSV containing predictions downloaded | ☐ |
| Open GitHub repository | Source files and README visible | ☐ |

## 4. Reflection answers
1. **Prediction target and why G3 is excluded.** The target is whether a student is AT RISK: G3 < 10 is 1 (at risk), G3 ≥ 10 is 0. G3 is the final grade the target is built from, so using it as an input would be target leakage: the model would simply read the answer and look far more accurate than it really is, and it would be useless for real prediction because G3 is unknown in advance.

2. **Recall for the "At risk" class.** Recall is the share of truly at-risk students that the model correctly flags. A recall of, say, 0.84 means about 84% of at-risk students were caught and about 16% were missed. Missed students are the costly error here, since they get no extra support, so recall matters more than accuracy alone, especially when the at-risk class is the smaller one.

3. **Why performance may differ at your institution.** The model learned patterns from one set of Portuguese secondary-school students. Grading scales, curriculum, culture, family and social context, and the distribution of study time or absences may differ, so the learned relationships may not hold. The dataset is also small, so the estimates are noisy. The model should be re-validated, or retrained, on local data before any use.

4. **What a faculty member should do after an AT RISK prediction.** Treat it as a prompt to look closer, not a verdict. Review the student's actual records (grades, attendance), talk with the student, and offer support such as tutoring, consultation, or counseling referral. The prediction must not be the sole basis for any decision about the student.

## 5. Submission checklist
1. ☐ SAD mini-analysis (section 1 above)
2. ☐ train_model.py and app.py
3. ☐ requirements.txt and README.md
4. ☐ Model evaluation results (section 2, filled in)
5. ☐ Screenshot of the Streamlit dashboard and prediction results
6. ☐ GitHub repository URL
