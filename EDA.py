import pandas as pd
import matplotlib.pyplot as plt

# Names of the columns need to be chnaged accordingly to the dataset used
def analyze_gender_distribution(file_path):
    """
    Analyzes the gender distribution in a CSV dataset (prints a statement about it) and visualizes the results
    with a bar chart showing the proportion of each gender (saves the bar chart). Counts posts labeled as female/male, prints statement abou it.

    Parameters:
    - file_path (str): Path to the CSV file containing the dataset.

    Returns:
    - Nothing.
    """
    w_m_df = pd.read_csv(file_path)
    proportion = w_m_df['female'].value_counts(normalize=True)
    female_counts = w_m_df['female'].sum()
    male_counts = len(w_m_df) - female_counts
    print(proportion)
    print(f'The amount of posts labeled as Female:', female_counts)
    print(f'The amount of posts labeled as Male:', male_counts)
    
    # Creating a bar chart for gender distribution
    plt.figure(figsize=(6, 4))
    plt.rcParams.update({"font.family": "serif", "font.size": 12})
    bars = proportion.plot(kind='bar', color='#6495ED')
    plt.title('Gender Distribution in the Dataset')
    plt.ylabel('Proportion')
    plt.xlabel('Gender')
    plt.xticks(rotation=0)
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adding numerical labels on top of the bars
    for bar in bars.patches:
        plt.text(bar.get_x() + bar.get_width()/2, 
                 bar.get_height() - 0.05,  # Adjusted height for visibility
                 f'{bar.get_height():.2f}', 
                 ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.savefig('Gender_Distribution_figure.png')
    plt.show()

    return


def gender_word_presence(file_path):
    """
    Analyzes the percentage of posts containing the words 'woman' and 'man'
    in a text dataset (prints a statement about it) and visualizes the results with a bar chart (saves the bar chart).

    Parameters:
    - file_path (str): Path to the CSV file containing the dataset.

    Returns:
    - Nothing.
    """
    w_m_df = pd.read_csv(file_path)

    if 'post' not in w_m_df.columns:
        raise ValueError("The dataset must contain a 'post' column.")

    w_m_df['contains_woman'] = w_m_df['post'].str.contains(r'\bwoman\b', case=False, regex=True)
    w_m_df['contains_man'] = w_m_df['post'].str.contains(r'\bman\b', case=False, regex=True)

    # Calculate the percentage of posts containing each word
    woman_percentage = w_m_df['contains_woman'].mean() * 100
    man_percentage = w_m_df['contains_man'].mean() * 100
    print(f"Percentage of posts containing 'woman': {woman_percentage:.2f}%")
    print(f"Percentage of posts containing 'man': {man_percentage:.2f}%")

    word_presence = {
        'Woman': woman_percentage,
        'Man': man_percentage
    }
    # Plot the results as a bar chart
    plt.figure(figsize=(6, 4))
    plt.rcParams.update({"font.family": "serif", "font.size": 12})
    bars = plt.bar(word_presence.keys(), word_presence.values(), color='#6495ED')
    plt.title("Presence of words 'woman' and 'man' in the Dataset")
    plt.ylabel('Percentage of Posts')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adding numerical labels on the bars
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 5,
                 f"{bar.get_height():.2f}%", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('Word_Presence_figure.png')
    plt.show()

    return

analyze_gender_distribution('gender.csv')
gender_word_presence('gender.csv')
