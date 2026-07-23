# iOS app

Requires macOS + Xcode 15+.

    brew install xcodegen
    cd ios
    xcodegen generate
    open SquashLineCalling.xcodeproj

Tests: `xcodebuild test -scheme SquashLineCalling -destination 'platform=iOS Simulator,name=iPhone 15'`

- `Sources/` — SwiftUI app (Play = native record; Matches/Coach = webview).
- `Model/` — drop `BallDetector.mlpackage` here (see `MODEL.md`).
- Server origin: `Sources/Config.swift`.
