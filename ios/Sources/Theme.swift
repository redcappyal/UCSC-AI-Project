import SwiftUI

/// DESIGN.md dark-theme tokens. IN/OUT colors are for verdicts ONLY;
/// the accent is always paired with black text.
enum Theme {
    static let bg = Color(hex: 0x000000)
    static let surface = Color(hex: 0x1C1C1E)
    static let line = Color(hex: 0x26262A)
    static let dim = Color(hex: 0x98989F)
    static let text = Color.white
    static let accentBg = Color(hex: 0xFFD60A)
    static let accentText = Color.black
    static let inCall = Color(hex: 0x2ECC5E)
    static let outCall = Color(hex: 0xE03A2F)
    static let unknown = Color(hex: 0xC7C7CC)
}

extension Color {
    init(hex: UInt32) {
        self.init(
            red: Double((hex >> 16) & 0xFF) / 255,
            green: Double((hex >> 8) & 0xFF) / 255,
            blue: Double(hex & 0xFF) / 255)
    }
}
