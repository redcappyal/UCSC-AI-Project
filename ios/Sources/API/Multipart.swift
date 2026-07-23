import Foundation

enum Multipart {
    static func body(boundary: String, fields: [(String, String)],
                     fileField: String, filename: String,
                     contentType: String, fileData: Data) -> Data {
        var data = header(boundary: boundary, fields: fields, fileField: fileField,
                          filename: filename, contentType: contentType)
        data.append(fileData)
        data.append(trailer(boundary: boundary))
        return data
    }

    /// Streams the multipart body to a temp file so a long rally is never
    /// held in RAM (the in-memory path peaks at ~2x the video size). Caller
    /// uploads with URLSession.upload(for:fromFile:) and deletes the file.
    static func writeBody(boundary: String, fields: [(String, String)],
                          fileField: String, filename: String,
                          contentType: String, fileURL: URL) throws -> URL {
        let bodyURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("upload-\(UUID().uuidString).multipart")
        FileManager.default.createFile(atPath: bodyURL.path, contents: nil)
        let writer = try FileHandle(forWritingTo: bodyURL)
        defer { try? writer.close() }
        try writer.write(contentsOf: header(
            boundary: boundary, fields: fields, fileField: fileField,
            filename: filename, contentType: contentType))
        let reader = try FileHandle(forReadingFrom: fileURL)
        defer { try? reader.close() }
        while let chunk = try reader.read(upToCount: 1 << 20), !chunk.isEmpty {
            try writer.write(contentsOf: chunk)
        }
        try writer.write(contentsOf: trailer(boundary: boundary))
        return bodyURL
    }

    private static func header(boundary: String, fields: [(String, String)],
                               fileField: String, filename: String,
                               contentType: String) -> Data {
        var data = Data()
        func append(_ string: String) { data.append(Data(string.utf8)) }
        for (name, value) in fields {
            append("--\(boundary)\r\n")
            append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n\(value)\r\n")
        }
        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"\(fileField)\"; filename=\"\(filename)\"\r\n")
        append("Content-Type: \(contentType)\r\n\r\n")
        return data
    }

    private static func trailer(boundary: String) -> Data {
        Data("\r\n--\(boundary)--\r\n".utf8)
    }

    static func formURLEncoded(_ fields: [(String, String)]) -> String {
        var allowed = CharacterSet.alphanumerics
        allowed.insert(charactersIn: "-._~")
        return fields.map { name, value in
            let encoded = value.addingPercentEncoding(withAllowedCharacters: allowed) ?? value
            return "\(name)=\(encoded)"
        }.joined(separator: "&")
    }
}
