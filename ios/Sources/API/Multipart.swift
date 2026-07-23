import Foundation

enum Multipart {
    static func body(boundary: String, fields: [(String, String)],
                     fileField: String, filename: String,
                     contentType: String, fileData: Data) -> Data {
        var data = Data()
        func append(_ string: String) { data.append(Data(string.utf8)) }
        for (name, value) in fields {
            append("--\(boundary)\r\n")
            append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n\(value)\r\n")
        }
        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"\(fileField)\"; filename=\"\(filename)\"\r\n")
        append("Content-Type: \(contentType)\r\n\r\n")
        data.append(fileData)
        append("\r\n--\(boundary)--\r\n")
        return data
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
