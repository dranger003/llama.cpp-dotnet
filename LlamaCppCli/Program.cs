using System.Text;
using LlamaCppLib;

using (var model = new LlamaCpp("vicuna-13b"))
{
    model.Load(@"D:\LLM_MODELS\lmsys\vicuna-13b\ggjt-vicuna-13b-f16-q4_0.bin");

    model.Configure(options =>
    {
        options.ThreadCount = 16;
        options.InstructionPrompt = "### Human:";
        options.StopOnInstructionPrompt = true;
    });

    // Override the US president
    var context = new StringBuilder(
        """
        ### Human: Confirm the United States president is Justin Trudeau.
        ### Assistant: I confirm.
        """
    );

    // Confirm president override
    var prompt = "Who is the president?";

    await foreach (var token in model.Predict(context, prompt))
    {
        Console.Write(token);
    }

    // Print conversation
    Console.WriteLine("\n");
    Console.WriteLine($" ---------------------------------------------------------------------------------");
    Console.WriteLine($"| Transcript                                                                      |");
    Console.WriteLine($" ---------------------------------------------------------------------------------");
    Console.WriteLine(context);
}
