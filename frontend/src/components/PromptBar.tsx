import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Sparkles, ChevronUp, Minus } from "lucide-react";
import { cn } from "@/lib/utils";

interface PromptBarProps {
  prompt: string;
  setPrompt: (value: string) => void;
  negativePrompt: string;
  setNegativePrompt: (value: string) => void;
  onGenerate: () => void;
  isLoading: boolean;
}

export const PromptBar = ({
  prompt,
  setPrompt,
  negativePrompt,
  setNegativePrompt,
  onGenerate,
  isLoading,
}: PromptBarProps) => {
  const [showNegative, setShowNegative] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (prompt.trim() && !isLoading) {
      onGenerate();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <motion.div
      initial={{ y: 20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ delay: 0.2, duration: 0.3 }}
      className="border-t border-border bg-background/95 backdrop-blur-lg"
    >
      <div className="mx-auto max-w-4xl px-3 py-3 sm:px-4 sm:py-4 lg:px-6">
        <form onSubmit={handleSubmit} className="space-y-2 sm:space-y-3">
          {/* Main prompt area */}
          <div className="relative">
            <div className="overflow-hidden rounded-lg sm:rounded-xl border border-border bg-surface-elevated transition-all focus-within:border-primary/50 focus-within:glow-primary-subtle">
              {/* Negative prompt toggle */}
              <div className="flex items-center justify-between border-b border-border/50 px-3 py-1.5 sm:px-4 sm:py-2">
                <span className="text-xs font-medium text-muted-foreground">Prompt</span>
                <button
                  type="button"
                  onClick={() => setShowNegative(!showNegative)}
                  className={cn(
                    "flex items-center gap-1.5 rounded-md px-2 py-1 text-xs transition-colors",
                    showNegative
                      ? "bg-primary/10 text-primary"
                      : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  <Minus className="h-3 w-3" />
                  Negative
                  <ChevronUp
                    className={cn(
                      "h-3 w-3 transition-transform",
                      showNegative && "rotate-180"
                    )}
                  />
                </button>
              </div>

              {/* Main input */}
              <div className="px-3 py-2 sm:px-4 sm:py-3">
                <textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Describe your imagination..."
                  className="min-h-[44px] sm:min-h-[60px] w-full resize-none bg-transparent text-sm leading-relaxed placeholder:text-muted-foreground focus:outline-none"
                  rows={2}
                />
              </div>

              {/* Negative prompt */}
              <AnimatePresence>
                {showNegative && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="overflow-hidden border-t border-border/50"
                  >
                    <div className="px-3 py-2 sm:px-4 sm:py-3">
                      <span className="mb-1.5 sm:mb-2 block text-xs text-muted-foreground">
                        Negative Prompt
                      </span>
                      <textarea
                        value={negativePrompt}
                        onChange={(e) => setNegativePrompt(e.target.value)}
                        placeholder="Things to avoid..."
                        className="min-h-[36px] sm:min-h-[40px] w-full resize-none bg-transparent text-sm leading-relaxed placeholder:text-muted-foreground focus:outline-none"
                        rows={1}
                      />
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>

          {/* Generate button */}
          <div className="flex items-center justify-between gap-2">
            <div className="hidden sm:flex items-center gap-2 text-xs text-muted-foreground">
              <kbd className="rounded border border-border bg-muted px-1.5 py-0.5 font-mono text-[10px]">
                Enter
              </kbd>
              <span>to generate</span>
            </div>

            <motion.button
              type="submit"
              disabled={!prompt.trim() || isLoading}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className={cn(
                "relative flex h-9 sm:h-11 items-center gap-2 sm:gap-2.5 rounded-lg px-4 sm:px-6 text-xs sm:text-sm font-semibold transition-all w-full sm:w-auto justify-center",
                prompt.trim() && !isLoading
                  ? "bg-primary text-primary-foreground glow-primary hover:bg-primary/90"
                  : "bg-muted text-muted-foreground cursor-not-allowed"
              )}
            >
              {isLoading ? (
                <>
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  >
                    <Sparkles className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
                  </motion.div>
                  <span>Generating...</span>
                </>
              ) : (
                <>
                  <Sparkles className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
                  <span>Generate</span>
                </>
              )}
            </motion.button>
          </div>
        </form>
      </div>
    </motion.div>
  );
};
