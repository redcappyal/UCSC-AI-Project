# TestFlight

## One-time setup (do this on day 2 — it flushes out signing problems early)

1. Xcode → Settings → Accounts → add the Apple ID on the paid developer team.
2. `cd ios && xcodegen generate && open SquashLineCalling.xcodeproj`.
3. Target SquashLineCalling → Signing & Capabilities → select the Team.
   (Automatic signing; the bundle id app.crosscourt registers itself.)
4. App Store Connect → Apps → "+" → New App → iOS, name "Squash Line Calling",
   bundle id from step 3, SKU anything.
5. Select "Any iOS Device (arm64)" → Product → Archive → Distribute App →
   TestFlight & App Store → Upload.
6. App Store Connect → TestFlight tab → wait for processing (~10 min) →
   Internal Testing → "+" group "Camp" → add testers by Apple ID email.
   Internal testers need no Beta App Review.
7. Testers install via the TestFlight app invitation email.

## Every subsequent build

1. Bump build number (target → General → Build, or agvtool).
2. Product → Archive → Distribute → Upload. The Camp group auto-updates.

## Before the real demo build

- Set the deployed server origin in `Sources/Config.swift`.
- Confirm `ios/Model/BallDetector.mlpackage` is present (MODEL.md) so the
  live overlay works.
