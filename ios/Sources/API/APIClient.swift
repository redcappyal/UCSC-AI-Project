import Foundation

protocol APIClientProtocol: Sendable {
    func latestCalibration() async throws -> LatestCalibration
    func upload(videoURL: URL) async throws -> UploadResponse
    func startTrack(videoID: String, calibrationJSON: String,
                    duration: Double) async throws -> JobStatus
    func trackStatus(runID: String) async throws -> JobStatus
}

struct APIClient: APIClientProtocol {
    var baseURL: URL = Config.baseURL
    var session: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 15 * 60   // big rally uploads
        return URLSession(configuration: config)
    }()

    func latestCalibration() async throws -> LatestCalibration {
        let url = baseURL.appending(path: "api/calibration/latest")
        let (data, response) = try await session.data(from: url)
        if (response as? HTTPURLResponse)?.statusCode == 404 { throw APIError.noCalibration }
        try Self.checkHTTP(response, data: data)
        return try LatestCalibration(responseData: data)
    }

    func upload(videoURL: URL) async throws -> UploadResponse {
        let boundary = "slc-\(UUID().uuidString)"
        var request = URLRequest(url: baseURL.appending(path: "api/upload"))
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)",
                         forHTTPHeaderField: "Content-Type")
        let fileData = try Data(contentsOf: videoURL)   // demo rallies: tens of MB
        let body = Multipart.body(
            boundary: boundary, fields: [],
            fileField: "video_file", filename: videoURL.lastPathComponent,
            contentType: "video/mp4", fileData: fileData)
        let (data, response) = try await session.upload(for: request, from: body)
        try Self.checkHTTP(response, data: data)
        return try JSONDecoder().decode(UploadResponse.self, from: data)
    }

    func startTrack(videoID: String, calibrationJSON: String,
                    duration: Double) async throws -> JobStatus {
        var request = URLRequest(url: baseURL.appending(path: "api/track"))
        request.httpMethod = "POST"
        request.setValue("application/x-www-form-urlencoded",
                         forHTTPHeaderField: "Content-Type")
        let form = Multipart.formURLEncoded([
            ("video_id", videoID),
            ("calibration_json", calibrationJSON),
            ("start_time", "0"),
            ("end_time", String(duration)),
            ("frame_stride", "4"),
            ("inference_width", "960"),
            ("event_engine", ""),
            ("fusion_3d", ""),
        ])
        request.httpBody = Data(form.utf8)
        let (data, response) = try await session.data(for: request)
        try Self.checkHTTP(response, data: data)
        return try JSONDecoder().decode(JobStatus.self, from: data)
    }

    func trackStatus(runID: String) async throws -> JobStatus {
        let url = baseURL.appending(path: "api/track/status/\(runID)")
        var request = URLRequest(url: url)
        request.cachePolicy = .reloadIgnoringLocalCacheData
        let (data, response) = try await session.data(for: request)
        try Self.checkHTTP(response, data: data)
        return try JSONDecoder().decode(JobStatus.self, from: data)
    }

    private static func checkHTTP(_ response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse else { throw APIError.badResponse }
        guard (200..<300).contains(http.statusCode) else {
            let message = (try? JSONDecoder().decode(
                [String: String].self, from: data))?["error"]
            throw APIError.http(http.statusCode, message)
        }
    }
}
