# Onnx Secure Loader

## Overview
Onnx Secure Loader is a Python-based tool for securely scanning and validating ONNX models. It provides comprehensive security checks, including operator validation, metadata verification, and prevention of vulnerabilities such as Path Traversal.

---

## Libraries Used

### Essential Libraries:
- **Path**: Simplifies file path management.
- **typing**: For defining data types such as `Set`, `List`, and `Optional`.
- **onnx**: Manages ONNX model operations.
- **logging**: Manages log messages.
- **hashlib**: Creates hashes for file validation.

---

## Logging Setup
To configure logging for debugging and system monitoring:

```python
import logging

logger = logging.getLogger("OnnxSecureLoader")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
```
- **Log Level**: DEBUG (tracks detailed system messages).
- **Message Format**: Includes timestamp, log level, and module name.

---

## Default Configurations

### Allowed Operators
```python
DEFAULT_ALLOWED_OPERATORS: Set[str] = { ... }
```
Defines the list of operators allowed in the ONNX model.

---

## Custom Exceptions

```python
class OnnxSecurityException(Exception):
    pass

class OnnxValidationException(Exception):
    pass
```
- **OnnxSecurityException**: Identifies security issues.
- **OnnxValidationException**: Identifies model integrity issues.

---

## OnnxCustomScanner Class

### Initialization
```python
class OnnxCustomScanner:
    def __init__(self, file_path: Path, allowed_operators: Set[str], file_hash: str):
        ...
```
Initializes the scanner with the following parameters:
- `file_path`: Path to the ONNX model file.
- `allowed_operators`: Set of allowed operators.
- `file_hash`: Expected hash value of the file.

### Core Functions

#### Scan Function
```python
def scan(self) -> bool:
    ...
```
Performs a comprehensive security and integrity check on the model.

#### Verify File Hash
```python
def _verify_file_hash(self) -> None:
    ...
```
Validates the file's hash to ensure integrity.

#### Load Model
```python
def _load_model(self) -> ModelProto:
    ...
```
Loads the ONNX model and performs basic integrity checks.

#### Validate Operators
```python
def _validate_operators(self, model: ModelProto) -> None:
    ...
```
Ensures all operators in the model are from the allowed list.

#### Validate Metadata
```python
def _validate_metadata(self, model: ModelProto) -> None:
    ...
```
Verifies metadata for potential harmful characters.

#### Validate External Data Paths
```python
def _validate_external_data_paths(self, model: ModelProto) -> None:
    ...
```
Checks for Path Traversal vulnerabilities in external files.

#### Disarm Function
```python
def disarm(self) -> bool:
    ...
```
Neutralizes identified vulnerabilities and prepares a sanitized model.

#### Save Cleaned Model
```python
def save_cleaned_model(self, output_path: Path) -> None:
    ...
```
Saves the sanitized model to a new file.

---

## Usage
1. Import the required libraries.
2. Configure logging.
3. Initialize the `OnnxCustomScanner` class with appropriate parameters.
4. Use the `scan` method to validate the model.
5. If necessary, use the `disarm` method to sanitize the model and save it using `save_cleaned_model`.

---

## Example
```python
from pathlib import Path
from onnx import ModelProto

# Initialize scanner
scanner = OnnxCustomScanner(
    file_path=Path("model.onnx"),
    allowed_operators={"Add", "Mul", "Relu"},
    file_hash="abc123"
)

# Perform scan
if scanner.scan():
    print("Model passed all security checks.")
else:
    print("Model contains vulnerabilities.")

# Disarm and save cleaned model
scanner.disarm()
scanner.save_cleaned_model(Path("cleaned_model.onnx"))
