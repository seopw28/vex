## Sample Ratio Mismatch (SRM) in A/B Testing

### What is SRM?

Sample Ratio Mismatch (SRM) occurs in A/B testing 
when the actual sample sizes of the control and treatment groups deviate significantly from the expected ratio. 

For example, if you expect a 50/50 split between the control and treatment groups, 
but the actual split is 60/40, you have an SRM.


### Why is SRM Important?

SRM is a critical issue because it can invalidate the results of an A/B test. 
If the sample sizes are not as expected, 
it can introduce bias and affect the statistical power of the test. 
This can lead to incorrect conclusions about the effectiveness of the treatment.

### Causes of SRM

Several factors can cause SRM, including:
- **Randomization Issues**: Problems with the randomization process can lead to uneven sample sizes.
- **Technical Errors**: Bugs in the code or issues with the experiment setup can cause SRM.
- **User Behavior**: Differences in user behavior, such as drop-offs or different engagement levels, can also lead to SRM.

### Detecting SRM

To detect SRM, you can use statistical tests to compare the observed sample sizes with the expected sample sizes. 
One common method is to use a chi-squared test to check for significant deviations.

#### Example: Chi-Squared Test for SRM
