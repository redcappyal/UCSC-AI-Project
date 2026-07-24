import SwiftUI

/// Server-address override sheet (gear button on the Play tab). Persists to
/// UserDefaults so standalone launches keep the LAN address — the Xcode
/// scheme argument only applied to Xcode-initiated launches.
struct ServerSettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @AppStorage(Config.serverBaseKey) private var storedBase = ""
    @State private var address = ""
    @State private var check: CheckState = .idle

    private enum CheckState: Equatable {
        case idle, checking
        case reachable
        case failed(String)
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField(Config.defaultBase, text: $address)
                        .keyboardType(.URL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .foregroundStyle(Theme.text)
                } header: {
                    Text("Server address")
                } footer: {
                    Text("The Flask pipeline this phone uploads rallies to, e.g. http://192.168.1.20:5000. Leave empty for the default (\(Config.defaultBase)).")
                }

                Section {
                    Button {
                        Task { await testConnection() }
                    } label: {
                        HStack {
                            Text("Test connection")
                            Spacer()
                            switch check {
                            case .idle:
                                EmptyView()
                            case .checking:
                                ProgressView()
                            case .reachable:
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundStyle(Theme.inCall)
                            case .failed:
                                Image(systemName: "xmark.circle.fill")
                                    .foregroundStyle(Theme.outCall)
                            }
                        }
                    }
                    .disabled(check == .checking)
                    if case .failed(let message) = check {
                        Text(message)
                            .font(.footnote)
                            .foregroundStyle(Theme.dim)
                    }
                }
            }
            .scrollContentBackground(.hidden)
            .background(Theme.bg)
            .navigationTitle("Server")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        Config.setServerBase(address)
                        dismiss()
                    }
                    .fontWeight(.semibold)
                }
            }
        }
        .onAppear { address = storedBase }
        .onChange(of: address) { check = .idle }
    }

    private func testConnection() async {
        check = .checking
        let candidate = candidateURL()
        guard let url = candidate?.appending(path: "api/health") else {
            check = .failed("Not a valid address.")
            return
        }
        var request = URLRequest(url: url)
        request.timeoutInterval = 5
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            let status = (response as? HTTPURLResponse)?.statusCode ?? 0
            check = status == 200
                ? .reachable
                : .failed("Server answered with HTTP \(status).")
        } catch {
            check = .failed(error.localizedDescription)
        }
    }

    /// The URL the current field contents would resolve to, without saving.
    private func candidateURL() -> URL? {
        let trimmed = address.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return URL(string: Config.defaultBase) }
        let withScheme = trimmed.contains("://") ? trimmed : "http://" + trimmed
        guard let url = URL(string: withScheme), url.host() != nil else { return nil }
        return url
    }
}
