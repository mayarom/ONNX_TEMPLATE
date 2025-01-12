from pathlib import Path
from typing import Any, Optional, Set, List
import onnx
from onnx import defs, ModelProto
import os
import logging
import hashlib

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
DEFAULT_MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
DEFAULT_MAX_TENSOR_SIZE = 1e9

# ---------------------------------------------------------
# Custom Exceptions (Maya)
# ---------------------------------------------------------
class OnnxSecurityException(Exception):
    """Security exception for ONNX loading vulnerabilities."""

class OnnxValidationException(Exception):
    """Validation exception for model content issues."""

# ---------------------------------------------------------
# Main secure loader class (Maya)
# ---------------------------------------------------------
class OnnxSecureLoader:
    def __init__(
        self,
        allowed_operators: Optional[Set[str]] = None,
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
        max_tensor_size: float = DEFAULT_MAX_TENSOR_SIZE,
        known_hashes: Optional[List[str]] = None
    ):
        # Initialize allowed operators and security settings
        self.allowed_operators = allowed_operators or DEFAULT_ALLOWED_OPERATORS
        self.max_file_size = max_file_size
        self.max_tensor_size = max_tensor_size
        self.known_hashes = known_hashes

    def secure_load(self, file_path: str) -> ModelProto:
        """
        Securely load an ONNX model with comprehensive validation steps.

        :param file_path: Path to the ONNX model file.
        :return: Loaded and validated ONNX model.
        :raises: OnnxSecurityException, OnnxValidationException if validations fail.
        """
        self._verify_file_hash(file_path)  # Validate file integrity using hash.
        model = self._load_model(file_path)  # Load and validate the model structure.
        self._validate_operators(model)  # Ensure only allowed operators are present.
        self._validate_metadata(model)  # Check metadata for potential issues.
        self._validate_tensors(model)  # Validate tensor dimensions and sizes.
        self._validate_external_data_paths(model)  # Check external data paths for security risks.
        logger.info("** ONNX model passed secure validation successfully. **")
        return model

    def _verify_file_hash(self, file_path: str) -> None:
        """
        Verify the file's hash against known trusted hashes to ensure integrity.
        If no known hashes are provided, this check is skipped.

        :param file_path: Path to the ONNX model file.
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
        Load the ONNX model and perform basic structural validation.

        :param file_path: Path to the ONNX model file.
        :return: Loaded ONNX model.
        :raises: OnnxValidationException if the model structure is invalid.
        """
        try:
            model = onnx.load(file_path)
            onnx.checker.check_model(model)  # Perform basic structural checks.
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
        Validate that all operators in the model are part of the allowed whitelist.

        :param model: Loaded ONNX model.
        """
        for node in model.graph.node:
            if node.op_type not in self.allowed_operators:
                msg = (f"Disallowed operator '{node.op_type}' found in node '{node.name}'. "
                       f"Allowed operators are {self.allowed_operators}")
                logger.error(msg)
                raise OnnxValidationException(msg)
        logger.debug("Operators validation passed.")

    def _validate_metadata(self, model: ModelProto) -> None:
        """
        Check metadata properties for forbidden characters or suspicious content.

        :param model: Loaded ONNX model.
        """
        if not model.metadata_props:
            logger.debug("No metadata properties found.")
            return
        max_meta_length = 1024
        forbidden_chars = {"<", ">", "{", "}", "`"}
        for prop in model.metadata_props:
            value = prop.value
            if len(value) > max_meta_length:
                msg = (f"Metadata '{prop.key}' is too long "
                       f"({len(value)} chars). Potential malicious injection.")
                logger.error(msg)
                raise OnnxValidationException(msg)
            if any(ch in value for ch in forbidden_chars):
                msg = (f"Metadata '{prop.key}' contains forbidden characters. "
                       f"Potential scripting injection.")
                logger.error(msg)
                raise OnnxValidationException(msg)
        logger.debug("Metadata validation passed.")

    def _validate_tensors(self, model: ModelProto) -> None:
        """
        Validate tensor dimensions to prevent memory overload or malicious configurations.

        :param model: Loaded ONNX model.
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
                msg = (f"Tensor '{tensor.name}' is too large ({shape_size} elements). "
                       f"Exceeds max allowed {self.max_tensor_size}.")
                logger.error(msg)
                raise OnnxValidationException(msg)
        logger.debug("Tensors validation passed.")

    def _validate_external_data_paths(self, model: ModelProto) -> None:
        """
        Check external data paths for potential path traversal vulnerabilities.

        :param model: Loaded ONNX model.
        """
        for tensor in model.graph.initializer:
            for entry in tensor.external_data:
                if 'value' in entry:
                    path_str = entry['value']
                    if ".." in path_str or path_str.startswith("/") or path_str.startswith("\\"):
                        msg = (f"External data path '{path_str}' in tensor '{tensor.name}' "
                               f"may allow path traversal.")
                        logger.error(msg)
                        raise OnnxValidationException(msg)
        logger.debug("External data path validation passed.")

# ---------------------------------------------------------
# Custom vulnerability scanner (Roni)
# ---------------------------------------------------------
class ONNXScanner:
    def __init__(self, file_path: Path):
        # Initialize scanner with the given file path
        self.file_path = file_path
        self.scan_results = None
        self.cleaned_object = None

    def scan(self) -> bool:
        """
        Perform vulnerability scanning on the ONNX model.

        :return: True if scan succeeds, False otherwise.
        """
        print(f"Scanning file: {self.file_path}")
        try:
            model = onnx.load(self.file_path)
            onnx.checker.check_model(model)
            print("Model loaded and verified successfully.")
        except Exception as e:
            print(f"Error while loading the model: {e}")
            return False
        # Check each operator in the model for custom implementations
        for node in model.graph.node:
            if not defs.has(node.op_type):
                print(f"Custom operator detected: {node.op_type}, Node name: {node.name}")
            else:
                print(f"Standard operator: {node.op_type}, Node name: {node.name}")
        self.scan_results = {"example_vulnerability": "found"}
        return True

    def get_scan_results(self) -> Any:
        """
        Retrieve the results of the scan.

        :return: Scan results dictionary.
        """
        if self.scan_results is None:
            raise ValueError("Scan has not been run yet.")
        return self.scan_results

    def disarm(self) -> bool:
        """
        Disarm vulnerabilities detected in the model.

        :return: True if disarm succeeds.
        """
        print("Disarming detected vulnerabilities...")
        self.cleaned_object = "cleaned_object_example"
        return True

    def get_disarmed_object(self) -> Any:
        """
        Retrieve the disarmed version of the object.

        :return: Disarmed object.
        """
        if self.cleaned_object is None:
            raise ValueError("Object has not been disarmed yet.")
        return self.cleaned_object

    def save_as_new_file(self, output_path: Path) -> None:
        """
        Save the disarmed object to a new file.

        :param output_path: Path to save the new file.
        """
        if self.cleaned_object is None:
            raise ValueError("Object has not been disarmed yet.")
        with open(output_path, "w") as file:
            file.write(str(self.cleaned_object))
        print(f"Cleaned file saved to: {output_path}")
