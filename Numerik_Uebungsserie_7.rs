use std::simd::{f64x4, SimdFloat};

fn create_matrix_and_vector(n: usize) -> (Vec<f64>, Vec<f64>) {
    //"flache" Matrix als eindimensionaler Vektor
    let mut a = vec![0.0; n * n];
   
    let mut b = vec![1.0; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                a[i * n + j] = 4.0;
            } else {
                a[i * n + j] = -1.0 / ((i as f64 - j as f64).powi(2)); 
            }
        }
    }
    (a, b)
}

// Jacobi-Verfahren mit SIMD-Optimierung
fn jacobi_iteration_simd(
    a: &[f64],  // flache Matrix A
    b: &[f64],  
    n: usize,   
    tol: f64,   
    max_iter: usize, 
) -> (Vec<f64>, usize) {
    let mut x = vec![0.0; n];   
    let mut x_new = vec![0.0; n]; 
    let tol_simd = f64x4::splat(tol);

    for iter in 0..max_iter {
        for i in 0..n {
            let mut sum = 0.0;

            let row_start = i * n; // Startindex flachen Matrix
            let row = &a[row_start..row_start + n]; // Zeile i als Slice

            // SIMD-Berechnung in 4er Chunks
            let mut j = 0;
            while j + 4 <= n {
                let row_chunk = f64x4::from_slice(&row[j..j + 4]);
                let x_chunk = f64x4::from_slice(&x[j..j + 4]);
                sum += (row_chunk * x_chunk).reduce_sum();
                j += 4;
            }

            while j < n {
                sum += row[j] * x[j];
                j += 1;
            }
            sum -= row[i] * x[i];

            x_new[i] = (b[i] - sum) / row[i];
        }

        // Konvergenz prüfen
        let mut converged = true;
        for j in (0..n).step_by(4) {
            let x_new_chunk = f64x4::from_slice(&x_new[j..std::cmp::min(j + 4, n)]);
            let x_chunk = f64x4::from_slice(&x[j..std::cmp::min(j + 4, n)]);
            let diff = (x_new_chunk - x_chunk).abs();
            if diff.simd_ge(tol_simd).any() {
                converged = false;
                break;
            }
        }

        x.copy_from_slice(&x_new);

        if converged {
            return (x, iter + 1);
        }
    }

    (x, max_iter)
}

fn gauss_seidel_iteration(
    a: &[f64],  
    b: &[f64],  
    n: usize,  
    tol: f64,   
    max_iter: usize, 
) -> (Vec<f64>, usize) {
    let mut x = vec![0.0; n]; 
    for iter in 0..max_iter {
        let mut converged = true; // Konvergenz prüfen
        for i in 0..n {
            let mut sum = 0.0;
            let row_start = i * n; 
            for j in 0..i {
                sum += a[row_start + j] * x[j]; 
            }
            for j in (i + 1)..n {
                sum += a[row_start + j] * x[j];
            }
            let new_x = (b[i] - sum) / a[row_start + i]; 
            if (new_x - x[i]).abs() >= tol {
                converged = false;
            }
            x[i] = new_x;
        }
        if converged {
            return (x, iter + 1); // Lösung und Iterationen zurückgeben
        }
    }
    (x, max_iter) 
}

fn main() {
    let n = 1024; 
    let tol = 1e-6; 
    let max_iter = 10000; 

    // Matrix und Vektor erzeugen
    let (a, b) = create_matrix_and_vector(n);
    
    let (solution_jacobi, iter_jacobi) = jacobi_iteration_simd(&a, &b, n, tol, max_iter);
    println!(
        "Jacobi-Verfahren: {} Iterationen, erste 10 Werte: {:?}",
        iter_jacobi,
        &solution_jacobi[..10]
    );

    let (solution_gs, iter_gs) = gauss_seidel_iteration(&a, &b, n, tol, max_iter);
    println!(
        "Gauß-Seidel-Verfahren: {} Iterationen, erste 10 Werte: {:?}",
        iter_gs,
&solution_gs[..10]
    );
}
