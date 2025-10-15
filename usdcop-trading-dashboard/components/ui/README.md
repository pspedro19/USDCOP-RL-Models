# Professional Trading Interface Components

This directory contains a comprehensive set of UI components designed to create a Bloomberg Terminal-level trading experience with professional-grade interactivity and accessibility features.

## 🎯 Overview

The Professional Trading Interface provides:

- **Command Palette** with fuzzy search and keyboard shortcuts
- **Smart Crosshair** with magnetic snap and comprehensive price info
- **Touch Gestures** for mobile professional interactions
- **Context Menus** with comprehensive right-click actions
- **Workspace Management** for multi-monitor support
- **Accessibility Features** meeting WCAG AAA standards
- **Help System** with interactive tutorials

## 📦 Components

### 1. CommandPalette (`command-palette.tsx`)

A professional command palette with fuzzy search capabilities.

**Features:**
- Fuzzy search with Fuse.js
- Recent commands and favorites
- Keyboard navigation (↑↓ navigate, ↵ select, Esc close)
- Categories and shortcuts display
- Customizable command actions

**Usage:**
```tsx
import { CommandPalette } from '@/components/ui/command-palette';

const customCommands = [
  {
    id: 'buy-order',
    title: 'Place Buy Order',
    subtitle: 'Open long position',
    category: 'Trading',
    icon: TrendingUp,
    action: () => console.log('Buy order'),
    keywords: ['buy', 'long', 'order'],
    shortcut: 'B',
  },
];

<CommandPalette
  commands={customCommands}
  onCommandExecute={(command) => console.log('Executed:', command)}
/>
```

### 2. SmartCrosshair (`smart-crosshair.tsx`)

An intelligent crosshair with magnetic snapping to OHLC prices.

**Features:**
- Magnetic snap to OHLC values
- Comprehensive price tooltip
- Distance and percentage measurements
- Copy price functionality
- Smooth animations

**Usage:**
```tsx
import { SmartCrosshair } from '@/components/ui/smart-crosshair';

const [crosshairPosition, setCrosshairPosition] = useState({
  x: 0, y: 0, price: 0, time: 0, visible: false
});

<SmartCrosshair
  position={crosshairPosition}
  data={chartData}
  snapToPrice={true}
  showTooltip={true}
  onPriceCopy={(price) => console.log('Copied:', price)}
/>
```

### 3. TradingContextMenu (`trading-context-menu.tsx`)

Professional context menu system with comprehensive trading actions.

**Features:**
- Hierarchical menu structure
- Trading, drawing, and chart actions
- Keyboard shortcuts display
- Danger actions highlighting
- Custom menu items support

**Usage:**
```tsx
import { TradingContextMenu } from '@/components/ui/trading-context-menu';

<TradingContextMenu
  items={customMenuItems}
  onItemSelect={(item) => console.log('Selected:', item)}
>
  <div>Right-click me!</div>
</TradingContextMenu>
```

### 4. HelpSystem (`help-system.tsx`)

Comprehensive help system with tutorials and shortcuts guide.

**Features:**
- Interactive tutorials with step-by-step guidance
- Searchable help topics
- Comprehensive keyboard shortcuts reference
- Progress tracking
- Multiple difficulty levels

**Usage:**
```tsx
import { HelpSystem } from '@/components/ui/help-system';

<HelpSystem
  open={helpOpen}
  onOpenChange={setHelpOpen}
  defaultTab="tutorials"
/>
```

### 5. ProfessionalTradingInterface (`professional-trading-interface.tsx`)

The main integration component that brings everything together.

**Features:**
- Integrates all components seamlessly
- Touch and mouse gesture support
- Accessibility features
- Workspace management
- Professional visual design

**Usage:**
```tsx
import { ProfessionalTradingInterface } from '@/components/ui/professional-trading-interface';

<ProfessionalTradingInterface
  chartData={data}
  onTimeframeChange={(tf) => console.log('Timeframe:', tf)}
  onDrawingTool={(tool) => console.log('Tool:', tool)}
  onTradingAction={(action) => console.log('Trading:', action)}
  onChartAction={(action) => console.log('Chart:', action)}
>
  <YourChartComponent />
</ProfessionalTradingInterface>
```

## 🎣 Hooks

### 1. useKeyboardShortcuts

Comprehensive keyboard shortcuts system.

**Shortcuts:**
- **Timeframes:** 1, 5, H, D (1min, 5min, 1hour, 1day)
- **Trading:** B (buy), S (sell), X (close), Cmd+C (cancel)
- **Drawing:** T (trendline), F (fibonacci), R (rectangle), L (line)
- **Chart:** Cmd+S (save), Cmd+Z (undo), Space (pan), +/- (zoom)
- **System:** Cmd+K (command palette), Cmd+/ (help), F11 (fullscreen)

### 2. useTouchGestures

Professional touch gesture handling for mobile devices.

**Gestures:**
- Pinch to zoom with physics
- Pan with momentum
- Long press for context menu
- Double tap to reset view
- Three finger tap for tools

### 3. useWorkspaceManager

Multi-monitor workspace management system.

**Features:**
- Create, minimize, maximize, close windows
- Tile, cascade, stack arrangements
- Save and load workspace templates
- Multi-monitor support
- Fullscreen mode

### 4. useAccessibility

WCAG AAA accessibility features.

**Features:**
- Screen reader support
- High contrast mode
- Reduced motion preferences
- Font size adjustment
- Color blind support filters
- Keyboard navigation
- Focus indicators

## 🎨 Styling

Import the professional UI styles:

```tsx
import '@/styles/professional-ui.css';
```

**CSS Features:**
- High contrast mode
- Reduced motion support
- Focus indicators
- Screen reader styles
- Responsive design
- Touch target optimization

## ⌨️ Keyboard Shortcuts

| Shortcut | Action | Description |
|----------|--------|-------------|
| `Cmd+K` | Command Palette | Open quick actions |
| `Cmd+/` | Help System | Show help and tutorials |
| `1`, `5`, `H`, `D` | Timeframes | Switch chart timeframes |
| `B`, `S`, `X` | Trading | Buy, Sell, Close positions |
| `T`, `F`, `R`, `L` | Drawing Tools | Activate drawing tools |
| `Space` | Pan Mode | Activate pan tool |
| `+`, `-`, `0` | Zoom | Zoom in, out, reset |
| `Escape` | Cancel | Close dialogs/cancel actions |

## 📱 Touch Gestures

| Gesture | Action | Description |
|---------|--------|-------------|
| Pinch | Zoom | Scale chart view |
| Pan | Move | Move chart around |
| Long Press | Context Menu | Show options menu |
| Double Tap | Reset View | Return to default zoom |
| Three Finger Tap | Help | Open help system |

## ♿ Accessibility Features

- **Screen Reader Compatible**: ARIA labels and live regions
- **High Contrast Mode**: Enhanced visual contrast
- **Keyboard Navigation**: Full keyboard accessibility
- **Focus Management**: Visible focus indicators
- **Reduced Motion**: Respects user motion preferences
- **Font Scaling**: Adjustable text sizes
- **Color Blind Support**: Color blind friendly palettes
- **Skip Links**: Quick navigation for screen readers

## 🧪 Example Usage

See `components/examples/TradingTerminalExample.tsx` for a complete implementation example.

```tsx
import TradingTerminalExample from '@/components/examples/TradingTerminalExample';

export default function TradingPage() {
  return <TradingTerminalExample />;
}
```

## 🔧 Customization

### Custom Commands

```tsx
const customCommands: CommandItem[] = [
  {
    id: 'custom-action',
    title: 'Custom Action',
    subtitle: 'My custom functionality',
    category: 'Custom',
    icon: MyIcon,
    action: () => myCustomFunction(),
    keywords: ['custom', 'action'],
    shortcut: 'Ctrl+Shift+C',
  },
];
```

### Custom Context Menu

```tsx
const customMenuItems: ContextMenuItem[] = [
  {
    id: 'my-action',
    label: 'My Action',
    icon: MyIcon,
    action: () => console.log('Custom action'),
    submenu: [
      // Sub-menu items...
    ],
  },
];
```

### Accessibility Settings

```tsx
const { updateSetting, settings } = useAccessibility();

// Toggle high contrast
updateSetting('highContrastMode', !settings.highContrastMode);

// Change font size
updateSetting('fontSize', 'large');
```

## 🚀 Performance

- Optimized animations with framer-motion
- Efficient gesture handling with @use-gesture/react
- Fuzzy search with Fuse.js
- Virtual scrolling for large lists
- Minimal re-renders with React.memo
- GPU-accelerated animations

## 🌐 Browser Support

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

**Mobile:**
- iOS Safari 13+
- Chrome Mobile 80+
- Samsung Internet 12+

## 📝 Dependencies

Required packages:
- `cmdk` - Command palette
- `@use-gesture/react` - Touch gestures
- `fuse.js` - Fuzzy search
- `react-hotkeys-hook` - Keyboard shortcuts
- `framer-motion` - Animations
- `@radix-ui/*` - Accessible primitives
- `react-hot-toast` - Notifications

## 🐛 Known Issues

1. **iOS Safari**: Some gesture events may not fire on certain iOS versions
2. **Firefox**: Focus indicators may not appear consistently
3. **High Contrast**: Some complex gradients may not display properly

## 🤝 Contributing

When adding new features:

1. Follow WCAG AAA accessibility guidelines
2. Add comprehensive TypeScript types
3. Include keyboard shortcuts where appropriate
4. Test with screen readers
5. Add documentation and examples
6. Test on mobile devices

## 📄 License

Part of the USD/COP Trading Dashboard project.