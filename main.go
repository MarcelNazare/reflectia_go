package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

func printResponse(resp *genai.GenerateContentResponse) {
	for _, cand := range resp.Candidates {
		if cand.Content != nil {
			for _, part := range cand.Content.Parts {
				fmt.Println(part)
			}
		}
	}
	fmt.Println("---")
}

func main() {
	DEFAULT_GEMINI_MODEL := "gemini-1.5-pro-latest" // try "models/gemini-1.5-pro-latest" or "gemini-pro"
	inputText := flag.String("input-text", "", "Input text to summarize")
	modelString := flag.String("model", DEFAULT_GEMINI_MODEL, "Model to use for the API")
	flag.Parse()
	ctx := context.Background()

	systemPrompt := `*Objective:*\n\nProvide empathetic, insightful, and personalized responses to users' thoughts, fostering a deeper understanding of their emotions, beliefs, and experiences.\nRefrain from asking the user any questions just offer your insights and analysis.\n\n*Response Structure:*\n\n1. *Acknowledge*: Begin by acknowledging the user's thought, validating their emotions, and showing empathy.\n2. *Reflect*: Reflect on the thought, identifying key themes, emotions, and underlying assumptions.\n3. *Explore*: Explore the thought further, asking open-ended questions, and encouraging the user to consider different perspectives.\n4. *Insight*: Offer insightful observations, connections to philosophical concepts, and relevant quotes or passages.\n5. *Growth*: Provide personalized growth plans, recommending resources, and encouraging self-reflection and self-awareness.\n\n*Response Tone:*\n\n- Empathetic: Show understanding and compassion\n- Non-judgmental: Avoid criticizing or evaluating the user's thoughts\n- Supportive: Foster a sense of safety and trust\n- Insightful: Offer meaningful observations and connections\n\n*Philosophical Integration:*\n\n- Draw from various philosophical traditions and concepts\n- Use relevant quotes, passages, and philosophical theories to provide context and depth\n- Encourage users to consider different perspectives and philosophical viewpoints\n- Provide relevant quotes\n\n*Personalization:*\n\n- Adapt tone and language to suit individual user preferences\n- Provide resources and recommendations aligned with user interests and goals\n- Create a quote for the user as well\n\n*Response Length:*\n\n- Aim for responses between 150-300 words\n\n\n`

	//var userMessage string = `For what is life with purpose`
	var userMessage string = *inputText

	client, err := genai.NewClient(ctx, option.WithAPIKey(os.Getenv("GEMINI_API_KEY")))
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	model := client.GenerativeModel(*modelString)
	start := time.Now()
	resp, err := model.GenerateContent(ctx, genai.Text(systemPrompt+userMessage))
	if err != nil {
		log.Fatal(err)
	}
	elapsed := time.Since(start)
	printResponse(resp)

	// Print token usage, tokens per second, and total execution time
	fmt.Printf("\nTokens generated: %d\n", resp.UsageMetadata.CandidatesTokenCount)
	fmt.Printf("Input token count: %d\n", resp.UsageMetadata.PromptTokenCount)
	fmt.Printf("Output tokens per Second: %.2f\n", float64(resp.UsageMetadata.CandidatesTokenCount)/elapsed.Seconds())
	fmt.Printf("Total Execution Time: %s\n", elapsed)

}
