# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial changelog file to track changes

### Fixed
- **Windows Compatibility**: Resolved encoding issues that caused crashes on Windows systems
  - Added explicit UTF-8 encoding for logging in `main.py`
  - Updated HTTP request handling in `agents/agent.py` to properly handle Unicode characters
  - Modified `agents/swarm.py` to use proper JSON serialization in API requests
  - Added proper Content-Type headers with charset=utf-8 for all API communications
  - Fixed issues with special characters in API responses and logging output

### Changed
- Improved cross-platform compatibility for logging and API communications
- Updated request handling to use proper JSON serialization throughout the codebase
- Enhanced error handling for character encoding issues

## [1.0.0] - 2025-07-22
### Added

