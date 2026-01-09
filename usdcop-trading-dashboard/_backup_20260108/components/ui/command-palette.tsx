"use client";

import * as React from "react";
import { Command } from "cmdk";
import { Search, Clock, Star, Zap, TrendingUp, BarChart3, Settings, HelpCircle } from "lucide-react";
import * as Dialog from "@radix-ui/react-dialog";
import { cn } from "@/lib/utils";
import Fuse from "fuse.js";
import { useHotkeys } from "react-hotkeys-hook";

interface CommandItem {
  id: string;
  title: string;
  subtitle?: string;
  category: string;
  icon: React.ComponentType<{ className?: string }>;
  action: () => void;
  keywords: string[];
  shortcut?: string;
  favorite?: boolean;
}

interface CommandPaletteProps {
  commands?: CommandItem[];
  onCommandExecute?: (command: CommandItem) => void;
  className?: string;
}

const defaultCommands: CommandItem[] = [
  // Chart Commands
  {
    id: "chart.timeframe.1m",
    title: "Switch to 1 Minute",
    subtitle: "Change chart timeframe",
    category: "Chart",
    icon: BarChart3,
    action: () => console.log("Switch to 1m"),
    keywords: ["1", "minute", "timeframe", "chart"],
    shortcut: "1",
  },
  {
    id: "chart.timeframe.5m",
    title: "Switch to 5 Minutes",
    subtitle: "Change chart timeframe",
    category: "Chart",
    icon: BarChart3,
    action: () => console.log("Switch to 5m"),
    keywords: ["5", "minute", "timeframe", "chart"],
    shortcut: "5",
  },
  {
    id: "chart.timeframe.1h",
    title: "Switch to 1 Hour",
    subtitle: "Change chart timeframe",
    category: "Chart",
    icon: BarChart3,
    action: () => console.log("Switch to 1h"),
    keywords: ["1", "hour", "timeframe", "chart"],
    shortcut: "H",
  },
  {
    id: "chart.timeframe.1d",
    title: "Switch to 1 Day",
    subtitle: "Change chart timeframe",
    category: "Chart",
    icon: BarChart3,
    action: () => console.log("Switch to 1d"),
    keywords: ["1", "day", "daily", "timeframe", "chart"],
    shortcut: "D",
  },
  // Trading Commands
  {
    id: "trading.buy",
    title: "Place Buy Order",
    subtitle: "Open long position",
    category: "Trading",
    icon: TrendingUp,
    action: () => console.log("Buy order"),
    keywords: ["buy", "long", "order", "trade"],
    shortcut: "B",
  },
  {
    id: "trading.sell",
    title: "Place Sell Order",
    subtitle: "Open short position",
    category: "Trading",
    icon: TrendingUp,
    action: () => console.log("Sell order"),
    keywords: ["sell", "short", "order", "trade"],
    shortcut: "S",
  },
  // Drawing Tools
  {
    id: "tools.trendline",
    title: "Trend Line Tool",
    subtitle: "Draw trend lines",
    category: "Tools",
    icon: Zap,
    action: () => console.log("Trend line tool"),
    keywords: ["trend", "line", "draw", "tool"],
    shortcut: "T",
  },
  {
    id: "tools.fibonacci",
    title: "Fibonacci Tool",
    subtitle: "Draw fibonacci retracements",
    category: "Tools",
    icon: Zap,
    action: () => console.log("Fibonacci tool"),
    keywords: ["fibonacci", "fib", "retracement", "draw", "tool"],
    shortcut: "F",
  },
  // Settings
  {
    id: "settings.preferences",
    title: "Open Preferences",
    subtitle: "Configure application settings",
    category: "Settings",
    icon: Settings,
    action: () => console.log("Open preferences"),
    keywords: ["settings", "preferences", "config", "options"],
  },
  {
    id: "help.shortcuts",
    title: "Keyboard Shortcuts",
    subtitle: "View all available shortcuts",
    category: "Help",
    icon: HelpCircle,
    action: () => console.log("Show shortcuts"),
    keywords: ["help", "shortcuts", "keyboard", "hotkeys"],
  },
];

const CommandPalette = React.forwardRef<
  HTMLDivElement,
  CommandPaletteProps
>(({ commands = defaultCommands, onCommandExecute, className }, ref) => {
  const [open, setOpen] = React.useState(false);
  const [search, setSearch] = React.useState("");
  const [recentCommands, setRecentCommands] = React.useState<string[]>([]);
  const [favorites, setFavorites] = React.useState<string[]>([]);

  // Initialize Fuse.js for fuzzy search
  const fuse = React.useMemo(() => {
    return new Fuse(commands, {
      keys: [
        { name: "title", weight: 0.5 },
        { name: "subtitle", weight: 0.3 },
        { name: "keywords", weight: 0.2 },
      ],
      threshold: 0.3,
      includeScore: true,
    });
  }, [commands]);

  // Filtered commands based on search
  const filteredCommands = React.useMemo(() => {
    if (!search.trim()) {
      return commands;
    }
    return fuse.search(search).map(result => result.item);
  }, [search, fuse, commands]);

  // Group commands by category
  const groupedCommands = React.useMemo(() => {
    const groups: Record<string, CommandItem[]> = {};

    // Add recent commands if no search
    if (!search.trim() && recentCommands.length > 0) {
      groups["Recent"] = recentCommands
        .map(id => commands.find(cmd => cmd.id === id))
        .filter((cmd): cmd is CommandItem => cmd !== undefined)
        .slice(0, 5);
    }

    // Add favorites if no search
    if (!search.trim() && favorites.length > 0) {
      groups["Favorites"] = favorites
        .map(id => commands.find(cmd => cmd.id === id))
        .filter((cmd): cmd is CommandItem => cmd !== undefined);
    }

    // Group filtered commands
    filteredCommands.forEach(command => {
      if (!groups[command.category]) {
        groups[command.category] = [];
      }
      groups[command.category].push(command);
    });

    return groups;
  }, [filteredCommands, search, recentCommands, favorites, commands]);

  // Execute command
  const executeCommand = React.useCallback((command: CommandItem) => {
    command.action();
    onCommandExecute?.(command);

    // Add to recent commands
    setRecentCommands(prev => {
      const filtered = prev.filter(id => id !== command.id);
      return [command.id, ...filtered].slice(0, 10);
    });

    setOpen(false);
    setSearch("");
  }, [onCommandExecute]);

  // Toggle favorite
  const toggleFavorite = React.useCallback((commandId: string, event: React.MouseEvent) => {
    event.stopPropagation();
    setFavorites(prev =>
      prev.includes(commandId)
        ? prev.filter(id => id !== commandId)
        : [...prev, commandId]
    );
  }, []);

  // Keyboard shortcuts
  useHotkeys("mod+k", (event) => {
    event.preventDefault();
    setOpen(true);
  }, { enableOnFormTags: true });

  useHotkeys("escape", () => {
    if (open) {
      setOpen(false);
      setSearch("");
    }
  }, { enableOnFormTags: true, enabled: open });

  return (
    <Dialog.Root open={open} onOpenChange={setOpen}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50" />
        <Dialog.Content
          ref={ref}
          className={cn(
            "fixed top-[20%] left-1/2 transform -translate-x-1/2 w-full max-w-2xl",
            "bg-gray-900 border border-gray-700 rounded-lg shadow-2xl z-50",
            "max-h-[60vh] overflow-hidden",
            className
          )}
        >
          <Command className="bg-transparent">
            <div className="flex items-center border-b border-gray-700 px-4">
              <Search className="h-4 w-4 text-gray-400 mr-3" />
              <Command.Input
                value={search}
                onValueChange={setSearch}
                placeholder="Search commands... (Cmd+K)"
                className="flex-1 bg-transparent border-0 outline-none text-white placeholder-gray-400 py-4"
                autoFocus
              />
              <kbd className="hidden sm:inline-flex h-5 px-1.5 items-center gap-1 text-xs text-gray-400 bg-gray-800 rounded">
                ⌘K
              </kbd>
            </div>

            <Command.List className="max-h-96 overflow-y-auto p-2">
              <Command.Empty className="py-8 text-center text-gray-400">
                No commands found.
              </Command.Empty>

              {Object.entries(groupedCommands).map(([category, categoryCommands]) => (
                <Command.Group key={category} heading={category}>
                  <div className="px-2 py-1.5 text-xs font-medium text-gray-400 uppercase tracking-wider">
                    {category}
                  </div>
                  {categoryCommands.map((command) => (
                    <Command.Item
                      key={command.id}
                      value={command.id}
                      onSelect={() => executeCommand(command)}
                      className="flex items-center justify-between px-3 py-3 rounded-md cursor-pointer hover:bg-gray-800 data-[selected=true]:bg-gray-800 group"
                    >
                      <div className="flex items-center space-x-3">
                        <command.icon className="h-4 w-4 text-gray-400" />
                        <div>
                          <div className="text-white font-medium">{command.title}</div>
                          {command.subtitle && (
                            <div className="text-sm text-gray-400">{command.subtitle}</div>
                          )}
                        </div>
                      </div>

                      <div className="flex items-center space-x-2">
                        <button
                          onClick={(e) => toggleFavorite(command.id, e)}
                          className="opacity-0 group-hover:opacity-100 transition-opacity"
                        >
                          <Star
                            className={cn(
                              "h-3 w-3",
                              favorites.includes(command.id)
                                ? "text-yellow-400 fill-current"
                                : "text-gray-400"
                            )}
                          />
                        </button>
                        {command.shortcut && (
                          <kbd className="px-2 py-1 text-xs text-gray-400 bg-gray-800 rounded">
                            {command.shortcut}
                          </kbd>
                        )}
                      </div>
                    </Command.Item>
                  ))}
                </Command.Group>
              ))}
            </Command.List>

            <div className="border-t border-gray-700 px-4 py-2 text-xs text-gray-400 flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <span className="flex items-center space-x-1">
                  <kbd className="px-1.5 py-0.5 bg-gray-800 rounded">↑↓</kbd>
                  <span>navigate</span>
                </span>
                <span className="flex items-center space-x-1">
                  <kbd className="px-1.5 py-0.5 bg-gray-800 rounded">↵</kbd>
                  <span>select</span>
                </span>
                <span className="flex items-center space-x-1">
                  <kbd className="px-1.5 py-0.5 bg-gray-800 rounded">esc</kbd>
                  <span>close</span>
                </span>
              </div>
              <div className="flex items-center space-x-1">
                <Clock className="h-3 w-3" />
                <span>{recentCommands.length} recent</span>
              </div>
            </div>
          </Command>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
});

CommandPalette.displayName = "CommandPalette";

export { CommandPalette, type CommandItem };