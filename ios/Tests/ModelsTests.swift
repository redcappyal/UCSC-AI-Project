import XCTest
@testable import SquashLineCalling

final class ModelsTests: XCTestCase {
    func decode<T: Decodable>(_ type: T.Type, _ json: String) throws -> T {
        try JSONDecoder().decode(type, from: Data(json.utf8))
    }

    func testUploadResponseDecodes() throws {
        let response = try decode(UploadResponse.self, #"""
        {"ok": true, "video_id": "abc123", "fps": 30.0,
         "frame_count": 900, "duration": 30.0}
        """#)
        XCTAssertEqual(response.videoID, "abc123")
        XCTAssertEqual(response.duration, 30.0)
    }

    func testJobStatusRunningDecodesWithoutHits() throws {
        let status = try decode(JobStatus.self, #"""
        {"ok": true, "status": "running", "run_id": "1753", "stage": "tracking",
         "progress": 41.5, "processed_frames": 100, "total_frames": 241,
         "message": "Tracking frames..."}
        """#)
        XCTAssertEqual(status.status, "running")
        XCTAssertNil(status.hits)
        XCTAssertEqual(status.progress, 41.5)
    }

    func testHitPresenceFlagsAndFrontWallFilter() throws {
        let status = try decode(JobStatus.self, #"""
        {"ok": true, "status": "complete", "run_id": "1753", "hits": [
          {"frame": 120, "timestamp_seconds": 4.0, "call": "IN",
           "margin_px": 3.5, "event_type": "wall",
           "target_zone": {"zone": 3}, "wall_diagram": {"x": 1.0, "y": 2.0}},
          {"frame": 300, "timestamp_seconds": 10.0, "call": "OUT",
           "margin_px": -2.0,
           "target_zone": {"zone": 1}, "wall_diagram": {"x": 3.0, "y": 4.0}},
          {"frame": 400, "timestamp_seconds": 13.3, "call": "UNKNOWN",
           "event_type": "floor"},
          {"frame": 500, "timestamp_seconds": 16.6, "call": "AUDIO",
           "event_type": "wall"},
          {"frame": 600, "timestamp_seconds": 20.0, "call": null,
           "event_type": "racket"}
        ]}
        """#)
        let hits = try XCTUnwrap(status.hits)
        XCTAssertEqual(hits.count, 5)
        XCTAssertTrue(hits[0].hasTargetZone && hits[0].hasWallDiagram)
        XCTAssertFalse(hits[2].hasTargetZone)
        // Server emits call:null for racket/floor/side_wall-classified hits;
        // decoding must survive it (the whole JobStatus decode used to fail).
        XCTAssertNil(hits[4].call)
        // Mirrors app.py front_wall_hits_from_payload: needs zone + diagram,
        // event_type in (null, wall, unknown). The floor hit and the
        // diagram-less AUDIO hit drop out.
        XCTAssertEqual(hits.frontWall.map(\.frame), [120, 300])
    }

    func testLatestCalibrationKeepsRawJSON() throws {
        let data = Data(#"{"ok": true, "run_id": "99", "calibration": {"lines": [1, 2]}}"#.utf8)
        let cal = try LatestCalibration(responseData: data)
        XCTAssertEqual(cal.runID, "99")
        // Round-trips as JSON (key order may differ) — it is re-posted verbatim
        // to /api/track, never interpreted by the app.
        let parsed = try JSONSerialization.jsonObject(with: Data(cal.calibrationJSON.utf8)) as? [String: Any]
        XCTAssertEqual(parsed?["lines"] as? [Int], [1, 2])
    }
}
