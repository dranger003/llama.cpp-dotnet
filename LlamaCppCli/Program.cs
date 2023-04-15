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

    // Initial context
    var context = new StringBuilder(
        """
        ### Human: You are a professor from MIT.
        ### Assistant: I confirm.
        """
    );

    var prompt = "Explain the correlation and distanciation of recurrent and recursive functions.";

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
