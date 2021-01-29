package main

import (
	"fmt"
	"math/rand"
	"time"
)

func main() {
	seed := int64(time.Now().UnixNano())
	num_samples := 36
	var listOfValues []int64
	for i := 0; i < num_samples; i++ {
		seed += int64(i)
		rand.Seed(seed)
		listOfValues = append(listOfValues, int64(rand.Intn(num_samples)))
	}
	fmt.Println(listOfValues)
}