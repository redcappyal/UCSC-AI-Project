import Foundation

enum Config {
    /// Deployed Flask origin (deploy/DEPLOY.md). Update before archiving.
    /// Any build can override at runtime via UserDefaults key "serverBase"
    /// (e.g. from Xcode scheme arguments) for LAN testing.
    static let defaultBase = "https://squash.example.com"

    static var baseURL: URL {
        if let raw = UserDefaults.standard.string(forKey: "serverBase"),
           let url = URL(string: raw) {
            return url
        }
        return URL(string: defaultBase)!
    }
}
