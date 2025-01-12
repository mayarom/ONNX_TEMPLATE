"""
OnnxSecureLoader & ONNXScanner

The code demonstrates secure loading of ONNX models and a simple vulnerability scanner.
Instead of calling the regular 'onnx.load(file_path)' naively, we use 'OnnxSecureLoader', 
which implements security checks. This helps prevent potential malicious or malformed 
ONNX models from harming the system.

Below, the file size validation code is commented out because it's planned for a future phase.
"""

from pathlib import Path
from typing import Any, Optional, Set, List
import os
import logging
import hashlib

import onnx
from onnx import defs, ModelProto

# ---------------------------------------------------------
# Basic logging setup (Maya)
# ---------------------------------------------------------
logger = logging.getLogger("OnnxSecureLoader")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ---------------------------------------------------------
# Default settings (Maya)
# ---------------------------------------------------------
DEFAULT_ALLOWED_OPERATORS: Set[str] = {
    "Relu", "Conv", "MaxPool", "Flatten", "Add", "Sub", "Mul",
    "Gemm", "BatchNormalization", "Dropout", "Shape", "Gather",
    "Unsqueeze"
}

# Note: max_file_size is defined here for future usage
# (file size checking is commented out below, for a later stage).
DEFAULT_MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
DEFAULT_MAX_TENSOR_SIZE = 1e9

# ---------------------------------------------------------
# Custom Exceptions (Maya)
# ---------------------------------------------------------
class OnnxSecurityException(Exception):
    """
    A general security exception for ONNX loading vulnerabilities.
    Raised when any security policy violation or suspicious activity is detected.
    """

class OnnxValidationException(Exception):
    """
    A validation exception for issues in the ONNX model content, 
    such as disallowed operators, suspicious metadata, or invalid structure.
    """

# ---------------------------------------------------------
# Main secure loader class (Maya)
# ---------------------------------------------------------
class OnnxSecureLoader:
    """
    A secure loader class for ONNX models. Instead of using 'onnx.load(file_path)' directly,
    use 'secure_load' to perform multiple security checks (hash, operator validation, etc.).
    """

    def __init__(
        self,
        allowed_operators: Optional[Set[str]] = None,
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
        max_tensor_size: float = DEFAULT_MAX_TENSOR_SIZE,
        known_hashes: Optional[List[str]] = None
    ):
        """
        Initialize the secure ONNX loader with configurable security parameters:
          - allowed_operators: a set of allowed operator names.
          - max_file_size: maximum file size limit (not currently used, 
                           see commented code below for a future stage).
          - max_tensor_size: maximum allowed number of elements in a tensor.
          - known_hashes: optional list of trusted hashes for integrity checking.
        """
        self.allowed_operators = allowed_operators or DEFAULT_ALLOWED_OPERATORS
        self.max_file_size = max_file_size      # For future use, see comment below
        self.max_tensor_size = max_tensor_size
        self.known_hashes = known_hashes

    def secure_load(self, file_path: str) -> ModelProto:
        """
        Main method for securely loading an ONNX model:
         1. (Commented out, planned for a later stage) Optional file size check.
         2. File hash verification against known/trusted hashes (if provided).
         3. Loading the model in a controlled way (_load_model).
         4. Operator validation: ensure only allowed operators are used.
         5. Metadata validation: check for suspicious strings or forbidden chars.
         6. Tensor validation: check for negative dims or excessively large tensors.
         7. External data path validation to prevent path traversal.

        :param file_path: Path to the ONNX model file.
        :return: Loaded and validated ONNX model.
        :raises OnnxSecurityException: if a security violation is detected.
        :raises OnnxValidationException: if a validation issue occurs (e.g., disallowed operator).
        """

        # ---------------------------------------------------------
        # The file size validation code is commented out for now,
        # as it might be used in a future development stage:
        # ---------------------------------------------------------
        # self._validate_file_size(file_path)

        # 2. Verify file hash (integrity check) if known_hashes are provided
        self._verify_file_hash(file_path)

        # 3. Load model with built-in ONNX checks
        model = self._load_model(file_path)

        # 4. Validate operators (allowed vs. disallowed)
        self._validate_operators(model)

        # 5. Validate metadata for suspicious or forbidden characters
        self._validate_metadata(model)

        # 6. Validate tensor dimensions and sizes
        self._validate_tensors(model)

        # 7. Validate external data paths for potential path traversal
        self._validate_external_data_paths(model)

        logger.info("** ONNX model passed secure validation successfully. **")
        return model

    # ---------------------------------------------------------
    # File size validation code (planned for a future stage):
    # ---------------------------------------------------------
    #
    # def _validate_file_size(self, file_path: str) -> None:
    #     """
    #     Validate that the ONNX file does not exceed the maximum allowed size.
    #     This function is commented out for now as it may be used in a future stage,
    #     where checking or limiting file size is required.
    #     """
    #     actual_size = os.path.getsize(file_path)
    #     if actual_size > self.max_file_size:
    #         msg = (
    #             f"File '{file_path}' size ({actual_size} bytes) exceeds the maximum "
    #             f"allowed size of {self.max_file_size} bytes."
    #         )
    #         logger.error(msg)
    #         raise OnnxSecurityException(msg)
    #     logger.debug(f"File size validation passed for {file_path}. Size: {actual_size} bytes.")

    def _verify_file_hash(self, file_path: str) -> None:
        """
        Verify the file's hash against a known trusted list (if provided).
        If known_hashes is empty or None, skip the check.
        """
        if not self.known_hashes:
            logger.debug("No known hashes provided; skipping hash verification.")
            return

        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        file_hash = sha256_hash.hexdigest()

        if file_hash not in self.known_hashes:
            msg = f"File hash {file_hash} is not in the list of known/trusted hashes."
            logger.error(msg)
            raise OnnxSecurityException(msg)
        logger.debug(f"Hash verification passed for {file_path}.")

    def _load_model(self, file_path: str) -> ModelProto:
        """
        A wrapper for onnx.load with additional structure checks using onnx.checker.
        This avoids relying on a naive 'onnx.load' call alone.
        """
        try:
            model = onnx.load(file_path)
            onnx.checker.check_model(model)
            logger.debug(f"Model loaded and basic ONNX check passed: {file_path}")
            return model
        except onnx.onnx_cpp2py_export.checker.ValidationError as e:
            msg = f"Failed ONNX validation: {e}"
            logger.error(msg)
            raise OnnxValidationException(msg) from e
        except Exception as e:
            msg = f"General error while loading ONNX model: {e}"
            logger.error(msg)
            raise OnnxSecurityException(msg) from e

    def _validate_operators(self, model: ModelProto) -> None:
        """
        Verify that every operator in the graph is in the allowed_operators set.
        Otherwise, raise OnnxValidationException.
        """
        for node in model.graph.node:
            if node.op_type not in self.allowed_operators:
                msg = (
                    f"Disallowed operator '{node.op_type}' found in node '{node.name}'. "
                    f"Allowed operators are {self.allowed_operators}"
                )
                logger.error(msg)
                raise OnnxValidationException(msg)
        logger.debug("Operators validation passed.")

    def _validate_metadata(self, model: ModelProto) -> None:
        """
        Check model metadata for suspicious or forbidden characters. This helps 
        prevent injection-like attacks or unexpected text in metadata fields.
        """
        if not model.metadata_props:
            logger.debug("No metadata properties found.")
            return

        max_meta_length = 1024
        forbidden_chars = {"<", ">", "{", "}", "`"}

        for prop in model.metadata_props:
            value = prop.value

            # Check length to avoid excessively large metadata
            if len(value) > max_meta_length:
                msg = (
                    f"Metadata '{prop.key}' is too long "
                    f"({len(value)} chars). Potential malicious injection."
                )
                logger.error(msg)
                raise OnnxValidationException(msg)

            # Check forbidden characters for potential scripting or injection
            if any(ch in value for ch in forbidden_chars):
                msg = (
                    f"Metadata '{prop.key}' contains forbidden characters. "
                    f"Potential scripting injection."
                )
                logger.error(msg)
                raise OnnxValidationException(msg)
        logger.debug("Metadata validation passed.")

    def _validate_tensors(self, model: ModelProto) -> None:
        """
        Ensure that tensor dimensions are valid and do not exceed the defined max_tensor_size.
        This prevents memory overload or malicious dimension settings.
        """
        for tensor in model.graph.initializer:
            shape_size = 1
            for dim in tensor.dims:
                if dim < 0:
                    msg = f"Tensor '{tensor.name}' has negative dimension ({dim})."
                    logger.error(msg)
                    raise OnnxValidationException(msg)
                shape_size *= dim if dim > 0 else 1

            if shape_size > self.max_tensor_size:
                msg = (
                    f"Tensor '{tensor.name}' is too large ({shape_size} elements). "
                    f"Exceeds max allowed {self.max_tensor_size}."
                )
                logger.error(msg)
                raise OnnxValidationException(msg)
        logger.debug("Tensors validation passed.")

    def _validate_external_data_paths(self, model: ModelProto) -> None:
        """
        Check for potential path traversal in external data references.
        If an external data entry has '..' or an absolute path, 
        raise OnnxValidationException as it may allow unauthorized file access.
        """
        for tensor in model.graph.initializer:
            for entry in tensor.external_data:
                if 'value' in entry:
                    path_str = entry['value']
                    if ".." in path_str or path_str.startswith("/") or path_str.startswith("\\"):
                        msg = (
                            f"External data path '{path_str}' in tensor '{tensor.name}' "
                            f"may allow path traversal."
                        )
                        logger.error(msg)
                        raise OnnxValidationException(msg)
        logger.debug("External data path validation passed.")


# -----------------------------------------------------------------------------------
# Custom vulnerability scanner: Large tensors & malicious patterns in metadata (Roni)
# -----------------------------------------------------------------------------------
class ONNXScanner:
    """
    A simple vulnerability scanner for ONNX models.
    Unlike OnnxSecureLoader, here we still do a direct onnx.load call,
    but we add logic to detect large tensors, custom (unknown) operators,
    and suspicious metadata patterns.
    """

    def __init__(self, file_path: Path):
        """
        Initialize the scanner with:
         - file_path: path to the ONNX file.
         - tensor_size_threshold: threshold for what is considered a 'large' tensor.
         - suspicious_metadata_patterns: list of patterns to detect in metadata.
         - scan_results: a dictionary storing any findings.
         - cleaned_object: placeholder if a 'cleaned' version of the model is needed.
        """
        self.file_path = file_path
        self.tensor_size_threshold = 1e6
        self.suspicious_metadata_patterns: List[str] = []
        self.scan_results: dict = {}
        self.cleaned_object = None  

    def scan(self) -> bool:
        """
        Perform a scanning procedure:
         1. Directly load the model via onnx.load (less secure than OnnxSecureLoader).
         2. Run onnx.checker.check_model for basic structural integrity.
         3. Check each initializer's size vs. tensor_size_threshold.
         4. Identify custom ops (those not in defs.has).
         5. Check metadata for suspicious patterns from suspicious_metadata_patterns.

        :return: True if the scan completes (even if warnings are found),
                 False if loading/verification fails.
        """
        print(f"Scanning file: {self.file_path}")

        # Load the model without extra security checks - demonstration only
        try:
            model = onnx.load(self.file_path)
            onnx.checker.check_model(model)
            print("Model loaded and verified successfully.")
        except Exception as e:
            print(f"Error while loading the model: {e}")
            return False

        # Check initializers for large tensors
        for initializer in model.graph.initializer:
            print(f"Tensor: {initializer.name}")
            data_type = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.get(initializer.data_type, "Unknown")
            print(f"  Data Type: {data_type}")
            print(f"  Shape: {initializer.dims}")

            size = 1
            for dim in initializer.dims:
                size *= dim
            print(f"  Size: {size} elements")

            if size > self.tensor_size_threshold:
                warning_msg = (
                    f"Warning: Tensor '{initializer.name}' has {size} elements, "
                    f"which might cause resource exhaustion."
                )
                print(warning_msg)
                self.scan_results.setdefault("large_tensors", []).append({
                    "tensor_name": initializer.name,
                    "size": size
                })

        # Identify custom (non-standard) operators
        for node in model.graph.node:
            if not defs.has(node.op_type):
                print(f"Custom operator detected: {node.op_type}, Node: {node.name}")
                self.scan_results.setdefault("custom_ops", []).append({
                    "node_name": node.name,
                    "op_type": node.op_type
                })
            else:
                print(f"Standard operator: {node.op_type}, Node: {node.name}")

        # Scan metadata for suspicious patterns
        suspicious_metadata = self._check_metadata_for_malicious_patterns(model)
        if suspicious_metadata:
            print("Suspicious metadata detected:")
            for entry in suspicious_metadata:
                print(
                    f"  - Key: '{entry['metadata_key']}', "
                    f"Value: '{entry['metadata_value']}', "
                    f"Pattern: '{entry['found_pattern']}'"
                )
            self.scan_results["suspicious_metadata"] = suspicious_metadata

        # Example: marking a general vulnerability or scan conclusion
        self.scan_results["example_vulnerability"] = "found"

        return True

    def _check_metadata_for_malicious_patterns(self, model: ModelProto) -> List[dict]:
        """
        Inspect metadata for user-defined suspicious patterns (e.g. 'script:', 'exec', etc.)
        Patterns are listed in self.suspicious_metadata_patterns.
        """
        suspicious_entries = []

        if model.metadata_props:
            for prop in model.metadata_props:
                key_lower = prop.key.lower()
                value_lower = prop.value.lower()

                for pattern in self.suspicious_metadata_patterns:
                    if pattern in key_lower or pattern in value_lower:
                        suspicious_entries.append({
                            "metadata_key": prop.key,
                            "metadata_value": prop.value,
                            "found_pattern": pattern
                        })

        return suspicious_entries

    def get_scan_results(self) -> dict:
        """
        Return the accumulated scan results.
        Raises ValueError if the scan has not been run or no results were recorded.
        """
        if not self.scan_results:
            raise ValueError("Scan has not been run yet or no results recorded.")
        return self.scan_results
