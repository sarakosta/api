def classify_by_degree(degree_series, specialists_path, generalists_path, low_thresh=1, high_thresh=99):
    """
    Classify species by degree and save specialists and generalists in separate CSV files.

    Parameters:
    - degree_series: pd.Series with species names as index and degrees as values
    - specialists_path: filepath to save specialists CSV
    - generalists_path: filepath to save generalists CSV
    - low_thresh: percentile cutoff for specialists (default 25)
    - high_thresh: percentile cutoff for generalists (default 75)

    Returns:
    - classification: pd.Series with all classifications
    """

    low = np.percentile(degree_series, low_thresh)
    high = np.percentile(degree_series, high_thresh)

    def classify(k):
        if k <= low:
            return "specialist"
        elif k >= high:
            return "generalist"
        else:
            return "intermediate"

    classification = degree_series.apply(classify)

    # Save specialists
    specialists = classification[classification == "specialist"].reset_index()
    specialists.columns = ['species', 'classification']
    specialists.to_csv(specialists_path, index=False)

    # Save generalists
    generalists = classification[classification == "generalist"].reset_index()
    generalists.columns = ['species', 'classification']
    generalists.to_csv(generalists_path, index=False)

    return classification


classification_c = classify_by_degree(all_degrees_c, "specialists_controlled.txt", "generalists_controlled.txt")
classification_r = classify_by_degree(all_degrees_r, "specialists_restored.txt", "generalists_restored.txt")