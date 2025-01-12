# Onnx Secure Loader

## סקירה כללית
Onnx Secure Loader הוא כלי מבוסס Python לבדיקת אבטחה ותקינות של מודלים בפורמט ONNX. הכלי מספק בדיקות אבטחה מקיפות, כולל ולידציה של אופרטורים, בדיקות מטא-נתונים, ומניעת פגיעויות כמו Path Traversal.

---

## ספריות בשימוש

### ספריות חיוניות:
- **Path**: לניהול נתיבי קבצים בצורה פשוטה.
- **typing**: להגדרת סוגי נתונים כמו `Set`, `List`, ו-`Optional`.
- **onnx**: לניהול פעולות על מודלים בפורמט ONNX.
- **logging**: לניהול הודעות לוגים.
- **hashlib**: ליצירת `hash` לאימות קבצים.

---

## הגדרת לוגים
להגדרת לוגים לצורכי דיבוג ומעקב אחר פעולות המערכת:

```python
import logging

logger = logging.getLogger("OnnxSecureLoader")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
```
- **רמת לוגים**: DEBUG (למעקב אחר הודעות מפורטות).
- **פורמט הודעות**: כולל זמן, רמת לוג ושם המודול.

---

## הגדרות ברירת מחדל

### אופרטורים מותרים
```python
DEFAULT_ALLOWED_OPERATORS: Set[str] = { ... }
```
מגדיר את רשימת האופרטורים המותרים במודל ONNX.

---

## חריגות מותאמות אישית

```python
class OnnxSecurityException(Exception):
    pass

class OnnxValidationException(Exception):
    pass
```
- **OnnxSecurityException**: מזהה בעיות אבטחה.
- **OnnxValidationException**: מזהה בעיות תקינות במודל.

---

## מחלקת OnnxCustomScanner

### אתחול
```python
class OnnxCustomScanner:
    def __init__(self, file_path: Path, allowed_operators: Set[str], file_hash: str):
        ...
```
מאתחלת את הסורק עם הפרמטרים הבאים:
- `file_path`: נתיב לקובץ המודל.
- `allowed_operators`: סט של אופרטורים מותרים.
- `file_hash`: ערך ה-hash הצפוי של הקובץ.

### פונקציות עיקריות

#### פונקציה `scan`
```python
def scan(self) -> bool:
    ...
```
מבצעת בדיקות אבטחה ותקינות מקיפות על המודל.

#### אימות hash של הקובץ
```python
def _verify_file_hash(self) -> None:
    ...
```
מאמתת את ה-hash של הקובץ כדי להבטיח את תקינותו.

#### טעינת מודל
```python
def _load_model(self) -> ModelProto:
    ...
```
טוענת את המודל ומבצעת בדיקות תקינות בסיסיות.

#### ולידציה של אופרטורים
```python
def _validate_operators(self, model: ModelProto) -> None:
    ...
```
מוודאת שכל האופרטורים במודל נמצאים ברשימה המותרת.

#### ולידציה של מטא-נתונים
```python
def _validate_metadata(self, model: ModelProto) -> None:
    ...
```
בודקת את המטא-נתונים כדי לוודא שאין בהם תווים מסוכנים.

#### בדיקת נתיבי קבצים חיצוניים
```python
def _validate_external_data_paths(self, model: ModelProto) -> None:
    ...
```
מוודאת שאין פגיעויות Path Traversal בקבצים חיצוניים.

#### פונקציה `disarm`
```python
def disarm(self) -> bool:
    ...
```
מנטרלת פגיעויות שהתגלו ומכינה מודל מנוטרל.

#### שמירת מודל מנוטרל
```python
def save_cleaned_model(self, output_path: Path) -> None:
    ...
```
שומרת את המודל המנוטרל לקובץ חדש.

---

## שימוש
1. ייבוא הספריות הנדרשות.
2. הגדרת לוגים.
3. אתחול מחלקת `OnnxCustomScanner` עם הפרמטרים המתאימים.
4. שימוש בפונקציה `scan` כדי לוודא את תקינות המודל.
5. במידת הצורך, שימוש בפונקציה `disarm` לנטרול פגיעויות ושמירת המודל באמצעות `save_cleaned_model`.

---

## דוגמה
```python
from pathlib import Path
from onnx import ModelProto

# אתחול הסורק
scanner = OnnxCustomScanner(
    file_path=Path("model.onnx"),
    allowed_operators={"Add", "Mul", "Relu"},
    file_hash="abc123"
)

# ביצוע סריקה
if scanner.scan():
    print("המודל עבר את כל בדיקות האבטחה.")
else:
    print("המודל מכיל פגיעויות.")

# ניטרול ושמירת מודל מנוטרל
scanner.disarm()
scanner.save_cleaned_model(Path("cleaned_model.onnx"))
```
