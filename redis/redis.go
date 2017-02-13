package redis

import (
	"log"

	"gopkg.in/redis.v5"
)

func Run() {
	client := redis.NewClient(&redis.Options{
		Addr: "localhost:6379",
	})

	pong, err := client.Ping().Result()

	log.Println(pong, err)

	pubsub, err := client.PSubscribe("iot:houses:*")
	if err != nil {
		panic(err)
	}
	defer pubsub.Close()

	err = client.Publish("iot:houses:house_1", "message to house 1").Err()
	if err != nil {
		panic(err)
	}

	for {
		msg, err := pubsub.ReceiveMessage()

		if err != nil {
			panic(err)
		}

		log.Println(msg.Channel, msg.Payload)
	}
}
