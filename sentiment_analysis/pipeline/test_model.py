from sentiment_analysis.pipeline.train_model import SentimentModel

if __name__ == "__main__":
    phrases = [
        "poetically states at one point in this movie that we `` do n't care about the truth",
        "poetically",
        "Must see summer blockbuster",
        "A comedy-drama of nearly epic proportions rooted in a sincere performance by the title character undergoing midlife crisis"
    ]
    model = SentimentModel(load_pretrained=True)
    for phrase in phrases:
        print(model.get_sentiment(phrase))
