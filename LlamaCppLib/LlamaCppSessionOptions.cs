namespace LlamaCppLib
{
    public class LlamaCppSessionOptions
    {
        /// <summary>
        /// The template to use when predicting.
        /// The tokens recognized are {prompt} and {context}, which are replaced with the provided strings when calling Predict().
        /// </summary>
        public string? Template { get; set; }
    }
}
