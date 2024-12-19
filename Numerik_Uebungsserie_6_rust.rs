use core::f64;
extern crate packed_simd;
use packed_simd::f64x4;

fn simd_dot_product(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = f64x4::splat(0.0); // SIMD-Register initialisieren

    //Vektoren in 4er Chunks aufteilen
    let chunks_a = a.chunks_exact(4);
    let chunks_b = b.chunks_exact(4);

    for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
        //mit 4er Slices Operationen paralellisieren
        let vec_a = f64x4::from_slice(chunk_a);
        let vec_b = f64x4::from_slice(chunk_b);

        sum += vec_a * vec_b;
    }

    sum.horizontal_sum() //Werte zusammenfassen innerhalb SIMD Register
}

fn normalize_vector(vector: &[f64]) -> Vec<f64> {
    let norm = simd_dot_product(vector, vector).sqrt();

    vector
        .iter() //Referenziterator erzeugen
        .map(|&xi| xi / norm.max(f64::EPSILON)) //per Referenz norm auf elemente des iterators anwenden
        .collect()
}

fn vector_norm(vector: &[f64]) -> f64 {
    simd_dot_product(vector, vector).sqrt()
}

fn householder(mut a: Vec<Vec<f64>>) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let m: usize = a.len();
    let n: usize = a[0].len();
    let mut q = vec![vec![0.0; m]; m]; //Nullmatrix initialisieren

    //Q als Einheitsmatrix initialisieren
    for i in 0..m {
        q[i][i] = 1.0;
    }

    for k in 0..n {
        let mut x = a[k..].iter().map(|row| row[k]).collect::<Vec<_>>(); //über Zeilen ab k-ter Zeile iterieren; iterator in vektor umwandeln

        let norm_x = vector_norm(&x);

        //Householder-Vektor berechnen
        x[0] += x[0].signum() * norm_x;

        let v: Vec<f64> = normalize_vector(&x);

        for j in k..n {
            let a_col = a[k..].iter().map(|row| row[j]).collect::<Vec<_>>(); //
            let dot = simd_dot_product(&v, &a_col);

            for (idx, row) in a[k..].iter_mut().enumerate() {
                row[j] -= 2.0 * dot * v[idx];
            }
        }

        for j in 0..m {
            let q_row = q[j][k..].to_vec(); //k-te Zeile von Q kopieren
            let dot_q = simd_dot_product(&q_row, &v);
            for idx in 0..v.len() {
                q[j][k + idx] -= 2.0 * dot_q * v[idx];
            }
        }
    }

    (a, q)
}

fn matrix_multiply(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let m = a.len();
    let n = b[0].len();
    let mut result = vec![vec![0.0; n]; m];

    for i in 0..m {
        for j in 0..n {
            result[i][j] = a[i]
                .iter()
                .zip(b.iter().map(|row| row[j]))
                .map(|(&ai, bj)| ai * bj)
                .sum();
        }
    }

    result
}

fn are_equal(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>, epsilon: f64) -> bool {
    a.iter().zip(b.iter()).all(|(row_a, row_b)| {
        row_a
            .iter()
            .zip(row_b.iter())
            .all(|(&ai, &bi)| (ai - bi).abs() < epsilon)
    })
}

fn main() {
    let matrix: [[f64; 4]; 4] = [
        [5.0, 4.0, 1.0, 1.0],
        [4.0, 5.0, 1.0, 1.0],
        [1.0, 1.0, 4.0, 2.0],
        [1.0, 1.0, 2.0, 4.0],
    ];

    let a = matrix
        .iter()
        .map(|&row| row.to_vec())
        .collect::<Vec<Vec<f64>>>();

    let (r, q) = householder(a);

    // Ausgabe der Matrizen R und Q
    println!("Matrix Q:");
    for row in q.iter() {
        println!("{:?}", row);
    }

    println!("\nMatrix R:");
    for row in r.iter() {
        println!("{:?}", row);
    }
}
