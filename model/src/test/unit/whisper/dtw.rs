//! DTW alignment unit tests.

use crate::whisper::dtw;

#[test]
fn dtw_diagonal_path() {
    // Identity cost matrix: diagonal path [0,0], [1,1], ..., [n-1,n-1]
    let n = 5;
    let mut cost = vec![0.0f32; n * n];
    for i in 0..n {
        cost[i * n + i] = -1.0; // diagonal is cheapest
    }

    let (text, time) = dtw::dtw(&cost, n, n);
    assert_eq!(text.len(), time.len());
    assert_eq!(text.len(), 2 * n - 1); // DTW path length for diagonal

    // Path should end at (n-1, n-1) and start at (0, 0)
    assert_eq!(*text.first().unwrap(), 0);
    assert_eq!(*time.first().unwrap(), 0);
    assert_eq!(*text.last().unwrap(), n - 1);
    assert_eq!(*time.last().unwrap(), n - 1);
}

#[test]
fn dtw_asymmetric() {
    // 2 text tokens, 5 audio frames. Cost favors repeating text tokens.
    let n_rows = 2;
    let n_cols = 5;
    let cost = vec![
        -1.0, -1.0, -1.0, 0.0, 0.0, // token 0 aligns to frames 0-2
        0.0, 0.0, 0.0, -1.0, -1.0, // token 1 aligns to frames 3-4
    ];

    let (text, time) = dtw::dtw(&cost, n_rows, n_cols);
    assert_eq!(text.len(), time.len());
    // Path must visit all rows and columns
    assert!(text.contains(&0));
    assert!(text.contains(&1));
    assert!(time.contains(&0));
    assert!(time.contains(&4));
}

#[test]
fn median_filter_simple() {
    // Input: 1×5, filter_width=3, reflect padding
    let data = vec![1.0f32, 3.0, 2.0, 5.0, 4.0];
    let out = dtw::median_filter(&data, 1, 5, 3);
    // j=0: reflect(-1)=idx 1 → [3,1,3] → median 3
    // j=1: [1,3,2] → median 2
    // j=2: [3,2,5] → median 3
    // j=3: [2,5,4] → median 4
    // j=4: reflect(5)=2*4-5=3 → [5,4,5] → median 5
    assert_eq!(out, vec![3.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn median_filter_preserves_smooth() {
    let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let out = dtw::median_filter(&data, 1, 10, 3);
    // Already smooth — median should barely change it
    for i in 1..9 {
        assert!((out[i] - data[i]).abs() < 1.0, "median filter changed smooth data at {i}");
    }
}
