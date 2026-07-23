// Stub so the target compiles until Task 10 replaces it.
import SwiftUI

struct ResultsView: View {
    let clip: FinishedClip
    var body: some View {
        Text("Recorded \(Int(clip.duration))s clip")
            .foregroundStyle(Theme.text)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Theme.bg)
    }
}
