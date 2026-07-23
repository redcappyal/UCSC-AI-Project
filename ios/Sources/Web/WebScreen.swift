import SwiftUI
import WebKit

/// The existing web product inside the native shell. Remote by design:
/// web fixes ship without an app update (spec decision).
struct WebScreen: View {
    let url: URL
    var showsClose = false
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ZStack(alignment: .topTrailing) {
            WebViewRepresentable(url: url).ignoresSafeArea(edges: .bottom)
            if showsClose {
                Button {
                    dismiss()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title)
                        .foregroundStyle(Theme.dim)
                        .padding(12)
                }
                .accessibilityLabel("Close review")
            }
        }
        .background(Theme.bg)
    }
}

private struct WebViewRepresentable: UIViewRepresentable {
    let url: URL

    func makeUIView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.allowsInlineMediaPlayback = true
        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.isOpaque = false
        webView.backgroundColor = .black
        #if DEBUG
        webView.isInspectable = true
        #endif
        webView.load(URLRequest(url: url))
        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {
        // Loaded once in makeUIView; SwiftUI re-renders must not reload the page.
    }
}
