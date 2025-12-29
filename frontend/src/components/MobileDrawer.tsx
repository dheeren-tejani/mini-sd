import { motion, AnimatePresence } from "framer-motion";
import { X, Dices, Activity, Hash, Maximize, Settings } from "lucide-react";
import { Slider } from "@/components/ui/slider";
import { cn } from "@/lib/utils";

interface MobileDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  steps: number;
  setSteps: (value: number) => void;
  cfg: number;
  setCfg: (value: number) => void;
  seed: number;
  setSeed: (value: number) => void;
  dimensions: "512x512" | "768x768";
  setDimensions: (value: "512x512" | "768x768") => void;
  isReady: boolean;
}

export const MobileDrawer = ({
  isOpen,
  onClose,
  steps,
  setSteps,
  cfg,
  setCfg,
  seed,
  setSeed,
  dimensions,
  setDimensions,
  isReady,
}: MobileDrawerProps) => {
  const randomizeSeed = () => {
    setSeed(Math.floor(Math.random() * 2147483647));
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm lg:hidden"
          />

          {/* Drawer */}
          <motion.div
            initial={{ x: "-100%" }}
            animate={{ x: 0 }}
            exit={{ x: "-100%" }}
            transition={{ type: "spring", damping: 25, stiffness: 300 }}
            className="fixed inset-y-0 left-0 z-50 w-[85vw] max-w-[300px] border-r border-border bg-background lg:hidden overflow-y-auto"
          >
            {/* Header */}
            <div className="flex items-center justify-between border-b border-border px-4 py-4">
              <div className="flex items-center gap-2">
                <Settings className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm font-medium">Parameters</span>
              </div>
              <button
                onClick={onClose}
                className="flex h-8 w-8 items-center justify-center rounded-md hover:bg-muted"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto px-4 py-6">
              <div className="space-y-8">
                {/* Steps */}
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                    <Activity className="h-3.5 w-3.5" />
                    <span>Steps</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <Slider
                      value={[steps]}
                      onValueChange={(v) => setSteps(v[0])}
                      min={1}
                      max={100}
                      step={1}
                      className="flex-1"
                    />
                    <span className="ml-4 w-10 text-right text-sm font-medium tabular-nums">
                      {steps}
                    </span>
                  </div>
                </div>

                {/* CFG */}
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                    <Hash className="h-3.5 w-3.5" />
                    <span>Guidance Scale</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <Slider
                      value={[cfg]}
                      onValueChange={(v) => setCfg(v[0])}
                      min={1}
                      max={20}
                      step={0.5}
                      className="flex-1"
                    />
                    <span className="ml-4 w-10 text-right text-sm font-medium tabular-nums">
                      {cfg.toFixed(1)}
                    </span>
                  </div>
                </div>

                {/* Seed */}
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                    <Dices className="h-3.5 w-3.5" />
                    <span>Seed</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <input
                      type="number"
                      value={seed}
                      onChange={(e) => setSeed(parseInt(e.target.value) || 0)}
                      className="flex-1 h-9 rounded-md border border-input bg-background px-3 text-sm tabular-nums focus:outline-none focus:ring-1 focus:ring-ring"
                    />
                    <button
                      onClick={randomizeSeed}
                      className="flex h-9 w-9 items-center justify-center rounded-md border border-input bg-secondary hover:bg-muted"
                    >
                      <Dices className="h-4 w-4 text-muted-foreground" />
                    </button>
                  </div>
                </div>

                {/* Dimensions */}
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                    <Maximize className="h-3.5 w-3.5" />
                    <span>Dimensions</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    {(["512x512", "768x768"] as const).map((dim) => (
                      <button
                        key={dim}
                        onClick={() => setDimensions(dim)}
                        className={cn(
                          "h-9 rounded-md border text-sm font-medium transition-all",
                          dimensions === dim
                            ? "border-primary bg-primary/10 text-primary"
                            : "border-input bg-secondary text-muted-foreground hover:bg-muted"
                        )}
                      >
                        {dim}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Footer */}
            <div className="border-t border-border px-4 py-4">
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <span
                  className={cn(
                    "h-2 w-2 rounded-full",
                    isReady ? "bg-emerald-500" : "bg-amber-500 animate-pulse"
                  )}
                />
                <span>Model: {isReady ? "Ready" : "Processing..."}</span>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};
