# Security

## Supported versions

Fixes come out as a new 3.x release, so keep to the latest one.

| Version | Supported |
|---|---|
| 3.x | Yes |
| Before 3.0 | No |

## Reporting a vulnerability

Please don't open a public issue. Email jakhon37@gmail.com instead, with:

- what the problem is, and what it lets someone do;
- how to reproduce it: the steps, and the file that triggers it if there is one;
- the Seamtag version (Help → About, or `seamtag --version`) and your system.

The fix comes out in a new release, noted in [CHANGELOG.md](CHANGELOG.md).

Worth reporting: a file Seamtag opens (an image, a label file, a COCO file, a YOLO dataset or a config file) that makes it run code, read or write files outside what you opened, or go online.

## What Seamtag does with your files

Seamtag works on files on your computer. It reads the images and label files in the folders you open, and writes label files, and the datasets you export, where you choose. Images, labels and settings aren't sent anywhere: there is no account, no telemetry and no update check.

It goes online only for these:

- **AI models**, when you agree to download one: from the question an AI tool asks the first time it needs its model, or from AI → AI models. The files come from release downloads on github.com, and YOLO-World's text encoder from Amazon S3 (`clip-as-service.s3.us-east-2.amazonaws.com`). Each file is checked against its SHA-256 before it's kept, in `~/.cache/osam`.
- **A spelling dictionary.** On Windows and Linux, the spell checker built into Electron downloads one for your system's language from Google's servers when Seamtag first starts. On macOS the system's own spell checker is used.
- **Links** in the app, such as Help → Documentation and the licences in AI → AI models, open in your web browser.

Pre-labelling runs the ONNX model you choose, so use models from sources you trust.
