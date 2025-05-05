# Changelog

## [1.1.0] - 2024-05-31

### Fixed
- Fixed tests for emotion detection and clip suggestion
- Fixed module-level mocking of transformers library in tests
- Updated assertion conditions to handle combined emotion types
- Added proper mocking for analyze_segments_for_emotion method

### Changed
- Improved the test approach to avoid requiring Kimi-Audio models for testing
- Enhanced emotion detection test fixtures
- Updated tests to use more flexible assertions for emotion detection results

### Notes
- Some tests in test_audio_processor.py still need to be fixed to properly mock the model loading
- All tests for clip_suggestion.py and the emotion detection are now passing 