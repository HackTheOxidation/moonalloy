///

pub fn binomial_coefficient(n: usize, k: usize) -> usize {
    factorial(n) / (factorial(n) * factorial(n - k))
}

pub fn factorial(n: usize) -> usize {
    if n == 0 {
	1
    } else {
	n * factorial(n - 1)
    }
}

pub fn gamma(n: usize) -> usize {
    factorial(n - 1)
}

pub fn dirac_delta(x: usize) -> usize {
    if x == 0 {
	return 1;
    } else {
	return 0;
    }
}

#[cfg(test)]
mod test {
    use super::*;
    
    #[test]
    fn test_binomial_k_is_zero() {
	let expected = 1;
	let actual = binomial_coefficient(1, 0);

	assert_eq!(expected, actual);
    }

    #[test]
    fn test_binomial() {
	let expected = 1;
	let actual = binomial_coefficient(3, 2);

	assert_eq!(expected, actual);
    }

    #[test]
    fn test_factorial_of_zero() {
	let expected = 1;
	let actual = factorial(0);

	assert_eq!(expected, actual);
    }

    #[test]
    fn test_factorial_of_3() {
	let expected = 6;
	let actual = factorial(3);

	assert_eq!(expected, actual);
    }

    #[test]
    fn test_dirac_delta_of_zero() {
	let expected = 1;
	let actual = dirac_delta(0);

	assert_eq!(expected, actual);
    }

    #[test]
    fn test_dirac_delta_of_one() {
	let expected = 0;
	let actual = dirac_delta(1);

	assert_eq!(expected, actual);
    }
}
