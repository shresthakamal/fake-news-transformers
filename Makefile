train:
	python -m fake_news.train

predict:
	python -m fake_news.predict

tensorboard:
	tensorboard --logdir="./fake_news/tensorboard/"

clean:
	rm -rf fake_news/tensorboard/*
	rm -rf fake_news/logs/*
