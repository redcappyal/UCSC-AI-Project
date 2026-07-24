import Foundation

enum Config {
    /// Deployed Flask origin (deploy/DEPLOY.md). Update before archiving.
    /// Any build can override at runtime via UserDefaults key "serverBase"
    /// (e.g. from Xcode scheme arguments) for LAN testing — plain-HTTP LAN
    /// hosts work because Info.plist sets ATS NSAllowsLocalNetworking.
    static let defaultBase = "https://squash.example.com"
    static let serverBaseKey = "serverBase"

    static var baseURL: URL {
        if let raw = UserDefaults.standard.string(forKey: serverBaseKey),
           let url = URL(string: raw), url.scheme != nil, url.host() != nil {
            return url
        }
        return URL(string: defaultBase)!
    }

    /// Persist an override (Server settings sheet). Empty/invalid input
    /// clears the override so baseURL falls back to defaultBase.
    static func setServerBase(_ raw: String) {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            UserDefaults.standard.removeObject(forKey: serverBaseKey)
            return
        }
        // Bare host:port is the common LAN case; assume plain HTTP for it.
        let withScheme = trimmed.contains("://") ? trimmed : "http://" + trimmed
        UserDefaults.standard.set(withScheme, forKey: serverBaseKey)
    }
}
