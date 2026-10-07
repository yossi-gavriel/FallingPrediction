# מסירת falls_ml 0.12.3

גרסת Phase 5 נשארת 2.2.0. לא נקראו נתוני מטופלים אמיתיים ולא שונו או אומנו מחדש המודלים והתוצאות ב־Windows.

- ענף: `codex/phase5-0.12.3-sqlite-lifecycle`
- Commit: `4574d8886c1787f451d2c621a20f866cd62be41f`
- חבילה: `falls_ml_phase5_0.12.3_mailsafe.zip`
- SHA-256: `6824e76d67e4803adaba38b1558cc5dfd208ba5dc199b941a28d3d307e043b86`
- המקור שנבדק: commit `d1e611c8d0cf750ec8e439775f4ff902d6387da5`, גרסה 0.12.2; החבילה המקורית אומתה לפי ה־SHA שנמסר וכל 162 קובצי המקור התאימו בדיוק.
- הענף הראשי וענף `phase5` המקורי לא שונו. כל 20 קובצי המקור של Phase 5 ומניפסטי ההגנה ההיסטוריים זהים למקור.

## התקלה והתיקון

ל־Optuna נמסרה כתובת SQLite כמחרוזת בכל יצירה וטעינה של study. כך נוצרו מנועי SQLAlchemy ומאגרי חיבורים נפרדים, שלא נסגרו במפורש. הפניות מה־study/session/pool יכלו להשאיר את קובץ SQLite פתוח גם לאחר חזרת הפונקציה. Windows סירב להעביר את הקובץ לארכיון וזרק WinError 32.

זו בעיית מחזור חיים בקוד ייצור משותף של Phase 2/3, שהתגלתה בבדיקה. היא עשויה להשפיע במיוחד על חידוש בתוך אותו תהליך או לאחר חריגה; תהליך שנסגר לגמרי משחרר את הידיות דרך מערכת ההפעלה. אין מכך מסקנה שהריצה הקלינית שהושלמה שגויה. Phase 5 משתמש ב־Optuna בזיכרון, והדשבורד אינו משתמש באף מסלול tuning.

ב־`src/falls_ml/phase2/xgb_tuning.py` כל הפעלת `run_study` מחזיקה storage מפורש אחד. ה־session מוסר ואז ה־engine נסגר ב־finally, גם אם בניית ה־storage נכשלת. סימון `TrialStore.fresh` נעשה רק לאחר שהארכוב והטיפול ביומן הצליחו. אין דילוג על Windows, השהיה, ניסיון חוזר עיוור או התעלמות מהחריגה. סדר ההצעות, הזרעים ותוצאות החידוש נשמרו.

עודכנו גרסת החבילה, בדיקות, תיעוד וחריגת hash מדויקת בשומרי המקור עבור התיקון המצומצם ובדיקת ההגנה הרלוונטית. קוד האימון והעיבוד המקדים של Phase 5 לא השתנה.

## הדשבורד

אפשר להפיק דשבורד לתיקייה הקיימת `C:\Users\Yosef.g9\Downloads\100k_falling_db_phase5_v3`, בתנאי שהיא ריצת Phase 5 2.x שלמה ותקינה והקלט המקורי התואם זמין. הוא נדרש לזיהוי לפי hash ולסריקת פרטיות מקומית לקריאה בלבד. לא נבדקו קבצי הריצה האמיתית מרחוק.

הפקודה הייעודית קוראת outer-OOF ותוצרי הסבר שמורים ומייצרת דוחות מצרפיים. היא אינה מאמנת LASSO/ENET/XGB, אינה מפעילה Optuna או פותחת SQLite, אינה טוענת מודלים לשם חיזוי ואינה משנה מודלים או OOF. הדבר הוכח בגרף הקריאות, חסימות דינמיות והשוואת bytes/זמני שינוי/רשימות קבצים. גם לפני התיקון, התקלה המדווחת לא הייתה במסלול הדשבורד.

החישוב הראשי נשאר דירוג בתוך כל outer fold והקצאה יחסית בין הקפלים. השוואת סיכונים מאוחדת נשארת תיאורית בלבד. HTML עצמאי ללא רשת ונתוני מטופלים ברמת שורה. הכתיבות הן לדוחות share, לגיבויי share ב־work/dashboard וליומן הדשבורד. אין להשתמש ב־`meuhedet-phase5 --report-only`, משום שמסלול זה משחזר גם תוצרי OOF מקומיים.

## ביקורת עיבוד מקדים

**MATERIAL_ISSUE_RETRAINING_SHOULD_BE_CONSIDERED**

הביקורת מבוססת על מקור Phase 5 המדויק, שזהה בעיבוד ובאימון בין 0.12.1 ל־0.12.2, ולא על קוד Phase 2 הישן. טבלת CSV כוללת 130 פיצ'רים ואת כל השדות המבוקשים.

חציון להשלמת חסרים, גילוי קטגוריות, הסרת קבועים/כפילויות ותקנון נלמדים מחדש בכל inner training וב־outer training. התקנון לכל העמודות הלינאריות שנשמרו, כולל בינריות, הוא `(x - mean_train) / SD_train`, עם `ddof=0`; הערכים מחושבים לאחר קידוד והשלמת חסרים. תאריכי לוח שנה גולמיים אינם נכנסים ל־X. XGB מקבל ערכים מספריים ו־NaN ללא תקנון/חציון. קודי עישון/השמנה מקודדים לקטגוריות במסלול הלינארי, אך נשארים מספריים ב־XGB ב־ALL; הם אינם חלק מ־NEW SAFE. MEFI מקודד thermometer במסלול הלינארי. CCI נשאר רציף ללא אימות מרווחי הקבוצות.

הממצא המרכזי: זכאות לפי AUROC מחושבת על כלל המדגם והתוצאות לפני חלוקת outer CV. לכן גם תוצאות ה־holdout משפיעות על זכאות הפיצ'רים. כיוון והיקף ההשפעה בריצה האמיתית אינם ידועים. גם גריד lambda נבנה מ־outer training כולו, כולל תוויות inner validation; תוויות outer holdout אינן משמשות בשלב זה.

| חשש קודם | סיווג ב־Phase 5 |
|---|---|
| תקנון שהיה מקונן נכון | NOT_APPLICABLE — לא היה פגם; גם כאן התקנון מקומי לאימון |
| חציון לפני inner CV | FIXED_IN_PHASE5 |
| fractional polynomials שנבחרו לפי תוצאה לפני inner CV | NOT_APPLICABLE |
| זכאות פיצ'רים לפני outer CV | PRESENT_IN_PHASE5 |
| הסרת יתירות לפני outer CV | FIXED_IN_PHASE5 |
| CCI רציף | PRESENT_IN_PHASE5 |
| קיבוץ מוקדם לפי דפוסי חסרים | NOT_APPLICABLE |

אין צורך באימון מחדש בגלל תיקון SQLite או לצורך הדשבורד. יש לבחון את רשומות הזכאות וההגדרות המתודולוגיות של הריצה האמיתית לפני החלטה על אימון נוסף. ביקורת מקור בלבד אינה מוכיחה שחייבים לאמן מחדש ואינה מוכיחה שהביצועים הקליניים תקפים.

## בדיקות

- Python 3.11 בסביבה נקייה עם גרסאות מה־locks הקיימים: הסוויטה המהירה בחבילה משוחזרת — **2436 passed, 1 skipped, 71 deselected**.
- החבילה הסופית: **389/389** קובצי מניפסט אומתו; **143 passed** בבדיקות lifecycle, resume המקורית, dashboard, capacity, חריגות מקור וסקריפטי handoff.
- Python 3.13: **15 passed** בבדיקות lifecycle/resume/הגנת המקור.
- בדיקת אינטגרציה של דשבורד על ריצת Phase 5 סינתטית שאומנה והושלמה: **1 passed**.
- הבדיקות מחזיקות הפניות אמיתיות לחיבורי DBAPI עם GC כבוי ומוודאות שאי אפשר עוד להפעיל SQL, ואז מעבירות/מחליפות מיד את הקובץ. נבדקו rollback journal וגם WAL; חריגות constructor/objective/tell/ledger; וחידוש ארבע פעמים עם תוצאות וזרעים זהים ויומן ללא כפילויות.
- הסוויטה הארוכה כולה אינה מדווחת כעוברת: הרצה נוספת הופסקה לאחר **1194 passed, 1 skipped, 1 failed**. הכשל הישן ב־EDA הוא כיול לוגיסטי שאינו מוגדר כשמודל מפיק סיכון קבוע. הוא שוחזר בחבילה המקורית 0.12.2 עם אותו CSV סינתטי, pepper ותבנית; שתי גרסאות עברו עם fixtures טריים. קוד המסלול נשאר ללא שינוי.
- לא הייתה מכונת Windows או Linux לבדיקת מערכת הפעלה בפועל. הבדיקות עברו ב־macOS; הבדיקות המצורפות מיועדות לבדוק את נעילת הקובץ גם ב־Windows.

גם שתי בדיקות האינטגרציה המשלימות עברו:

- `test_kills_inside_trial_validation_and_share_then_resume_are_identical` — Phase 2: **1 passed** (1527.89 שניות). שלוש עצירות יזומות בתוך ניסוי XGBoost, בעת validation ולפני פרסום share; חידוש הסתיים בטבלאות זהות לריצה רצופה, תוצרים שכבר נשמרו לא השתנו ולא נוצרה כפילות ביומן ניסויים.
- `test_kill_and_resume_give_identical_tables` — Phase 3: **1 passed**. עצירה וחידוש הסתיימו באותם hashes לטבלאות OOF, ablation, שחזור פיצ'רים ואימות כמו ריצה רצופה.

אלה הרצות CLI מלאות על fixtures סינתטיים חדשים בלבד. קוד המקור שנבדק בהן זהה לחבילה הסופית; השינויים המאוחרים לפני האריזה היו תיעוד בלבד.

## פקודות Windows — CMD

פתחו את ה־ZIP ב־Downloads. אימות checksum אפשר לבצע על ה־ZIP לפני חילוץ:

```bat
certutil -hashfile "%USERPROFILE%\Downloads\falls_ml_phase5_0.12.3_mailsafe.zip" SHA256
cd /d "%USERPROFILE%\Downloads\falls_ml_phase5_0.12.3"
py -3.11 RESTORE_FILES.py.txt
setup_windows.cmd
set PYTHONUTF8=1
.venv\Scripts\python.exe -c "import falls_ml; print(falls_ml.__version__)"
.venv\Scripts\python.exe -m pytest tests/unit/test_phase2_core.py::test_study_resumes_with_identical_suggestions tests/unit/test_xgb_storage_lifecycle.py tests/unit/test_phase5_dashboard.py tests/unit/test_phase5_capacity.py -q
.venv\Scripts\python.exe -m falls_ml meuhedet-phase5-dashboard --out "C:\Users\Yosef.g9\Downloads\100k_falling_db_phase5_v3"
start "" "C:\Users\Yosef.g9\Downloads\100k_falling_db_phase5_v3\share\PHASE5_OPERATING_DASHBOARD.html"
```

נדרשים `PACKAGE VERIFIED`, `INSTALLATION SUCCESSFUL`, גרסה 0.12.3 ובדיקות שעברו. הדשבורד אמור לסיים עם `DASHBOARD_COMPLETE`, `privacy_passed: true`, `units_unchanged: true`. אם הקלט המקורי אינו נמצא בשם המתוכנן ליד תיקיית הפלט או בתוכה, הוסיפו `--input` עם נתיבו האמיתי. אלה אינן פקודות אימון מחדש.

התיעוד המפורט נמצא בחבילה תחת `docs/phase5/SQLITE_LIFECYCLE_PATCH_0.12.3.md`, `DASHBOARD_SAFETY_AUDIT.md`, `PREPROCESSING_AUDIT_0.12.2.md` ו־`PREPROCESSING_FEATURE_AUDIT_0.12.2.csv`.
