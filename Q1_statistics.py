from typing import List


def prob_rain_more_than_n(p: List[float], n: int) -> float:
    """
    Calculate the probability of having at least `n` rainy days out of `N` days, 
    given a list of daily rain probabilities. 

    This function uses the **Poisson Binomial distribution** to compute the probability
    of having exactly `k` rainy days out of `N` days, where each day has its own 
    independent probability of rain. It then sums the probabilities for having at least 
    `n` rainy days.

    Parameters:
    -----------
    p : List[float]
        A list of `N` float values where each `p[i]` represents the probability 
        of rain on day `i` (0 <= p[i] <= 1).

    n : int
        The threshold number of rainy days. The function calculates the probability 
        of having at least `n` rainy days.

    Returns:
    --------
    float
        The probability of having at least `n` rainy days out of `N` days, 
        based on the provided daily rain probabilities using the Poisson Binomial distribution.

    Example:
    --------
    p = [0.4, 0.6, 0.3]
    n = 2
    result = prob_rain_more_than_n(p, n)
    print(result)  # Output will be the probability of having at least 2 rainy days
    """

    N = len(p)  # Number of days (365 in the typical case)

    # Initialize DP array where dp[k] is the probability of having exactly k rainy days
    dp = [0] * (N + 1)  # We can have between 0 and N rainy days
    dp[0] = 1  # Before considering any days, the probability of 0 rainy days is 1

    # Traverse each day's rain probability
    for prob in p:
        # Update the DP array backwards to avoid overwriting
        for k in range(N, 0, -1):  # Traverse from N down to 1
            dp[k] = dp[k] * (1 - prob) + dp[k - 1] * prob
        # For 0 rainy days, we just consider no rain on this day
        dp[0] = dp[0] * (1 - prob)

    # The result is the sum of probabilities of having at least n rainy days
    return sum(dp[i] for i in range(n, N + 1))


# Example usage:
p = [0.4, 0.6, 0.3]  # Example: rain probabilities for 3 days
n = 2  # We're interested in the probability of having at least 2 rainy days

result = prob_rain_more_than_n(p, n)
print(f"Probability of having at least {n} rainy days: {result}")
