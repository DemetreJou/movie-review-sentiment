from sentiment_analysis.train_model import get_sentiment

if __name__ == "__main__":
    phrases = [
        "poetically states at one point in this movie that we `` do n't care about the truth",
        "poetically",
        "Must see summer blockbuster",
        "A comedy-drama of nearly epic proportions rooted in a sincere performance by the title character undergoing midlife crisis"
    ]
    for phrase in phrases:
        print(get_sentiment(phrase))
