use super::cell::CellVertex;

/// Solver for a quadratic error function to position a vertex within a cell
#[derive(Copy, Clone, Debug, Default)]
pub struct QuadraticErrorSolver {
    /// A^T A term
    ata: nalgebra::Matrix3<f32>,

    /// A^T B term
    atb: nalgebra::Vector3<f32>,

    /// B^T B term
    btb: f32,

    /// Mass point of intersections is stored as XYZ / W, so that summing works
    mass_point: nalgebra::Vector4<f32>,
}

impl std::ops::AddAssign for QuadraticErrorSolver {
    fn add_assign(&mut self, rhs: Self) {
        self.ata += rhs.ata;
        self.atb += rhs.atb;
        self.btb += rhs.btb;
        self.mass_point += rhs.mass_point;
    }
}

impl QuadraticErrorSolver {
    pub fn new() -> Self {
        Self {
            ata: nalgebra::Matrix3::zeros(),
            atb: nalgebra::Vector3::zeros(),
            btb: 0.0,
            mass_point: nalgebra::Vector4::zeros(),
        }
    }

    #[cfg(test)]
    pub fn mass_point(&self) -> nalgebra::Vector4<f32> {
        self.mass_point
    }

    /// Adds a new intersection to the QEF
    ///
    /// `pos` is the position of the intersection and is accumulated in the mass
    /// point.  `grad` is the gradient at the surface, and is normalized in this
    /// function.
    pub fn add_intersection(
        &mut self,
        pos: nalgebra::Vector3<f32>,
        grad: nalgebra::Vector4<f32>,
    ) {
        // Add the position to the mass point
        self.mass_point += nalgebra::Vector4::new(pos.x, pos.y, pos.z, 1.0);
        
        // Following libfive's implementation, normalize the gradients
        // and handle invalid normals gracefully
        let norm_val = grad.xyz().norm();
        
        // Skip points with invalid normals (following libfive's approach)
        if norm_val <= 1e-12 || grad.xyz().iter().any(|v| !v.is_finite()) {
            return;
        }
        
        // Normalize the gradient vector
        let norm = grad.xyz() / norm_val;
        
        // Calculate dot product for QEF (considering distance value)
        let value = grad.w / norm_val;  // Adjust the value by the norm
        let b = norm.dot(&pos) - value;
        
        // Update QEF matrices
        self.ata += norm * norm.transpose();
        self.atb += norm * b;
        self.btb += b * b;
    }

    /// Solve the given QEF, minimizing towards the mass point
    ///
    /// Returns a vertex localized within the given cell, and adjusts the solver
    /// to increase the likelihood that the vertex is bounded in the cell.
    ///
    /// Also returns the QEF error as the second item in the tuple
    ///
    /// Sets the rank based on the eigenvalues, following libfive's approach:
    /// - rank 0: all eigenvalues are invalid, use the center point
    /// - rank 1: the first eigenvalue is valid, this is a planar feature
    /// - rank 2: the first two eigenvalues are valid, this is an edge
    /// - rank 3: all eigenvalues are valid, this is a corner
    pub fn solve(&self) -> (CellVertex<3>, f32, u8) {
        // This gets a little tricky; see
        // https://www.mattkeeter.com/projects/qef for a walkthrough of QEF math
        // and references to primary sources.
        let center = self.mass_point.xyz() / self.mass_point.w;
        let atb = self.atb - self.ata * center;

        let svd = nalgebra::linalg::SVD::new(self.ata, true, true);

        // nalgebra doesn't always actually order singular values (?!?)
        // https://github.com/dimforge/nalgebra/issues/1215
        let mut singular_values =
            svd.singular_values.data.0[0].map(ordered_float::OrderedFloat);
        singular_values.sort();
        singular_values.reverse();
        let singular_values = singular_values.map(|o| o.0);

        // Skip any eigenvalues that are small relative to the maximum
        // eigenvalue.  This is very much a tuned value (alas!).  If the value
        // is too small, then we incorrectly pick high-rank solutions, which may
        // shoot vertices out of their cells in near-planar situations.  If the
        // value is too large, then we incorrectly pick low-rank solutions,
        // which makes us less likely to snap to sharp features.
        //
        // For example, our cone test needs to use a rank-3 solver for
        // eigenvalues of [1.5633028, 1.430821, 0.0058764853] (a dynamic range
        // of 2e3); while the bear model needs to use a rank-2 solver for
        // eigenvalues of [2.87, 0.13, 5.64e-7] (a dynamic range of 10^7).  We
        // Use an adaptive cutoff strategy that's more aggressive for preserving sharp features
        // Base cutoff from libfive (0.1) as a starting point
        const BASE_EIGENVALUE_CUTOFF: f32 = 0.1;
        
        // For highly anisotropic features (where one eigenvalue is much larger than others)
        // we want to be more aggressive in preserving the sharp feature
        let ratio_01 = if singular_values[0] > 0.0 && singular_values[1] > 0.0 {
            singular_values[0] / singular_values[1]
        } else {
            1.0
        };
        
        let ratio_12 = if singular_values[1] > 0.0 && singular_values[2] > 0.0 {
            singular_values[1] / singular_values[2]
        } else {
            1.0
        };
        
        // If ratios are high, we have a strong feature (edge or corner)
        // Adjust cutoff to be more aggressive (lower) in these cases
        let cutoff = if ratio_01 > 10.0 || ratio_12 > 10.0 {
            BASE_EIGENVALUE_CUTOFF * 0.5
        } else {
            BASE_EIGENVALUE_CUTOFF
        };

        // Intuition about `rank`:
        // 0 => all eigenvalues are invalid (?!), use the center point
        // 1 => the first eigenvalue is valid, this must be planar
        // 2 => the first two eigenvalues are valid, this is a planar or an edge
        // 3 => all eigenvalues are valid, this is a planar, edge, or corner
        let rank = (0..3)
            .find(|i| singular_values[*i].abs() < cutoff)
            .unwrap_or(3);

        let epsilon = singular_values.get(rank).cloned().unwrap_or(0.0);
        let sol = svd.solve(&atb, epsilon);
        let pos = sol.map(|c| c + center).unwrap_or(center);
        // We'll clamp the error to a small > 0 value for ease of comparison
        let err = ((pos.transpose() * self.ata * pos
            - 2.0 * pos.transpose() * self.atb)[0]
            + self.btb)
            .max(1e-6);

        (CellVertex { pos }, err, rank as u8)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use nalgebra::{Vector3, Vector4};

    #[test]
    fn qef_rank2() {
        let mut q = QuadraticErrorSolver::new();
        q.add_intersection(
            Vector3::new(-0.5, -0.75, -0.75),
            Vector4::new(0.24, 0.12, 0.0, 0.0),
        );
        q.add_intersection(
            Vector3::new(-0.75, -1.0, -0.6),
            Vector4::new(0.0, 0.0, 0.31, 0.0),
        );
        q.add_intersection(
            Vector3::new(-0.50, -1.0, -0.6),
            Vector4::new(0.0, 0.0, 0.31, 0.0),
        );
        let (_out, err, _rank) = q.solve();
        assert_eq!(err, 1e-6);
    }

    #[test]
    fn qef_near_planar() {
        let mut q = QuadraticErrorSolver::new();
        q.add_intersection(
            Vector3::new(-0.5, -0.25, 0.4999981),
            Vector4::new(-0.66666776, -0.33333388, 0.66666526, -1.2516975e-6),
        );
        q.add_intersection(
            Vector3::new(-0.5, -0.25, 0.50),
            Vector4::new(-0.6666667, -0.33333334, 0.6666667, 0.0),
        );
        q.add_intersection(
            Vector3::new(-0.5, -0.25, 0.50),
            Vector4::new(-0.6666667, -0.33333334, 0.6666667, 0.0),
        );
        let (out, err, _rank) = q.solve();
        assert_eq!(err, 1e-6);
        let expected = Vector3::new(-0.5, -0.25, 0.5);
        assert!(
            (out.pos - expected).norm() < 1e-3,
            "expected {expected:?}, got {:?}",
            out.pos
        );
    }
}
