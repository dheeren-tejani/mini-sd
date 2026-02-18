import { motion } from "framer-motion";
import { Dices, Zap, Maximize, Hash, Activity } from "lucide-react";
import { Slider } from "@/components/ui/slider";
import { cn } from "@/lib/utils";

interface SidebarProps {
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

const ParameterSection = ({
  label,
  icon: Icon,
  children,
}: {
  label: string;
  icon: React.ElementType;
  children: React.ReactNode;
}) => (
  <div className="space-y-3">
    <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
      <Icon className="h-3.5 w-3.5" />
      <span>{label}</span>
    </div>
    {children}
  </div>
);

export const Sidebar = ({
  steps,
  setSteps,
  cfg,
  setCfg,
  seed,
  setSeed,
  dimensions,
  setDimensions,
  isReady,
}: SidebarProps) => {
  const randomizeSeed = () => {
    setSeed(Math.floor(Math.random() * 2147483647));
  };

  return (
    <motion.aside
      initial={{ x: -20, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className="hidden lg:flex w-[280px] xl:w-[300px] flex-col border-r border-border bg-sidebar"
    >
      {/* Header */}
      <div className="flex items-center gap-3 border-b border-border px-6 py-5">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10 glow-primary-subtle">
          <Zap className="h-4 w-4 text-primary" />
        </div>
        <div>
          <h1 className="text-base font-semibold tracking-tight">Toy SD</h1>
          <p className="text-xs text-muted-foreground">AI Image Studio</p>
        </div>
      </div>

      {/* Parameters */}
      <div className="flex-1 overflow-y-auto px-6 py-6">
        <div className="space-y-8">
          {/* Steps */}
          <ParameterSection label="Steps" icon={Activity}>
            <div className="space-y-3">
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
              <p className="text-xs text-muted-foreground">
                Higher steps = more detail, slower generation
              </p>
            </div>
          </ParameterSection>

          {/* Guidance Scale */}
          <ParameterSection label="Guidance Scale" icon={Hash}>
            <div className="space-y-3">
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
              <p className="text-xs text-muted-foreground">
                How closely to follow your prompt
              </p>
            </div>
          </ParameterSection>

          {/* Seed */}
          <ParameterSection label="Seed" icon={Dices}>
            <div className="flex items-center gap-2">
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(parseInt(e.target.value) || 0)}
                className="flex-1 h-9 rounded-md border border-input bg-background px-3 text-sm tabular-nums placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
                placeholder="Random seed"
              />
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={randomizeSeed}
                className="flex h-9 w-9 items-center justify-center rounded-md border border-input bg-secondary hover:bg-muted transition-colors"
                title="Randomize seed"
              >
                <Dices className="h-4 w-4 text-muted-foreground" />
              </motion.button>
            </div>
            <p className="text-xs text-muted-foreground">
              Same seed = reproducible results
            </p>
          </ParameterSection>

          {/* Dimensions */}
          <ParameterSection label="Dimensions" icon={Maximize}>
            <div className="grid grid-cols-2 gap-2">
              {(["512x512", "768x768"] as const).map((dim) => (
                <motion.button
                  key={dim}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => setDimensions(dim)}
                  className={cn(
                    "h-9 rounded-md border text-sm font-medium transition-all",
                    dimensions === dim
                      ? "border-primary bg-primary/10 text-primary glow-primary-subtle"
                      : "border-input bg-secondary text-muted-foreground hover:bg-muted hover:text-foreground"
                  )}
                >
                  {dim}
                </motion.button>
              ))}
            </div>
          </ParameterSection>
        </div>
      </div>

      {/* Footer */}
      <div className="border-t border-border px-6 py-4">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span
            className={cn(
              "h-2 w-2 rounded-full",
              isReady ? "bg-emerald-500 shadow-lg shadow-emerald-500/50" : "bg-amber-500 animate-pulse"
            )}
          />
          <span>Model Status: {isReady ? "Ready" : "Processing..."}</span>
        </div>
      </div>
    </motion.aside>
  );
};
