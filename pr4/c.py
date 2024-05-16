from scipy.stats import f_oneway

from a import iris_df, iris

# Check if 'target' column exists in the dataframe
if 'target' in iris_df.columns:
    # Grouping by target and calculating variance of each feature
    variance_by_target = iris_df.groupby('target').var()

    # Perform ANOVA for each feature
    for feature in iris['feature_names']:
        anova_result = f_oneway(
            *[iris_df[iris_df['target'] == i][feature] for i in range(len(iris.target_names))]
        )
        print(f"{feature} ANOVA:", anova_result)
else:
    print("No categorical variable found for ANOVA.")
