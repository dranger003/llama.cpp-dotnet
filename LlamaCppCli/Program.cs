using LlamaCppLib;

using (var model = new LlamaCpp("vicuna-13b-v1.1"))
{
    model.Load(@"D:\LLM_MODELS\lmsys\vicuna-13b-v1.1\ggml-vicuna-13b-v1.1-q4_0.bin");

    model.Configure(options =>
    {
        options.ThreadCount = 16;
        options.EndOfStreamToken = "</s>";
    });

    var session = model.NewSession("Conversation #1");

    session.Configure(options =>
    {
        options.InitialContext.AddRange(
            new[]
            {
                "A chat between a user and an assistant.",
                "USER: Hello!",
                "ASSISTANT: Hello!</s>",
                "USER: How are you?",
                "ASSISTANT: I am good.</s>",
            }
        );

        options.Roles.AddRange(new[] { "USER", "ASSISTANT", "ASSIST" });
    });

    var prompts = new[]
    {
        "USER: How many planets are there in the solar system?",
        "USER: Can you list the planets?",
        "USER: What is Vicuna 13B?",
        "USER: It is a large language model.",
    };

    Console.WriteLine(session.InitialContext.Aggregate((a, b) => $"{a}\n{b}"));

    foreach (var prompt in prompts)
    {
        Console.WriteLine(prompt);

        await foreach (var token in session.Predict(prompt))
            Console.Write(token);
    }

    // Print conversation
    Console.WriteLine();
    Console.WriteLine($" --------------------------------------------------------------------------------------------------");
    Console.WriteLine($"| Transcript                                                                                       |");
    Console.WriteLine($" --------------------------------------------------------------------------------------------------");

    foreach (var topic in session.Conversation)
        Console.WriteLine(topic);
}
