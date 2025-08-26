#!/usr/bin/env python3
"""
DLQ Tools
=========
Command-line tools for managing Dead Letter Queues.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.messaging.dlq import DLQInspector
from src.core.messaging.dlq import DLQManager
from src.core.messaging.dlq import RetryHandler, RetryPolicy

async def list_dlq_messages(inspector: DLQInspector, topic: str, limit: int, offset: int):
    """List messages in a DLQ topic."""
    print(f"\n📋 DLQ Messages for topic: {topic}")
    print("=" * 60)
    
    messages = await inspector.list_dlq_messages(topic, limit=limit, offset=offset)
    
    if not messages:
        print("No messages found in DLQ.")
        return
    
    for i, message in enumerate(messages, 1):
        print(f"\n📨 Message {i + offset}:")
        print(f"   Original Topic: {message.get('original_topic', 'Unknown')}")
        print(f"   Error: {message.get('error', 'Unknown')}")
        print(f"   Retry Count: {message.get('retry_count', 0)}")
        print(f"   First Failed: {message.get('first_failed', 'Unknown')}")
        print(f"   Last Failed: {message.get('last_failed', 'Unknown')}")
        print(f"   Correlation ID: {message.get('correlation_id', 'None')}")
        print(f"   Source Service: {message.get('source_service', 'Unknown')}")
        
        # Show additional context if available
        additional_context = message.get('additional_context', {})
        if additional_context:
            print(f"   Additional Context: {json.dumps(additional_context, indent=6)}")

async def show_dlq_summary(inspector: DLQInspector):
    """Show summary of all DLQ topics."""
    print("\n📊 DLQ Summary")
    print("=" * 60)
    
    summary = await inspector.get_dlq_summary()
    
    if not summary:
        print("No DLQ topics found.")
        return
    
    for topic, info in summary.items():
        print(f"\n📁 Topic: {topic}")
        print(f"   Message Count: {info.get('message_count', 0)}")
        
        oldest_age = info.get('oldest_message_age')
        if oldest_age is not None:
            oldest_hours = oldest_age / 3600
            print(f"   Oldest Message: {oldest_hours:.1f} hours ago")
        
        newest_age = info.get('newest_message_age')
        if newest_age is not None:
            newest_hours = newest_age / 3600
            print(f"   Newest Message: {newest_hours:.1f} hours ago")

async def replay_message(inspector: DLQInspector, topic: str, message_id: str, target_topic: str = None):
    """Replay a message from DLQ."""
    print(f"\n🔄 Replaying message {message_id} from {topic}")
    print("=" * 60)
    
    success = await inspector.replay_message(topic, message_id, target_topic)
    
    if success:
        target = target_topic or topic.replace('.dlq', '')
        print(f"✅ Message successfully replayed to {target}")
    else:
        print("❌ Failed to replay message")

async def purge_old_messages(inspector: DLQInspector, topic: str, days: int):
    """Purge old messages from DLQ."""
    print(f"\n🗑️  Purging messages older than {days} days from {topic}")
    print("=" * 60)
    
    purged_count = await inspector.purge_dlq_messages(topic, days)
    
    if purged_count > 0:
        print(f"✅ Purged {purged_count} old messages")
    else:
        print("ℹ️  No old messages found to purge")

async def analyze_patterns(inspector: DLQInspector, topic: str, limit: int):
    """Analyze patterns in DLQ messages."""
    print(f"\n🔍 Analyzing patterns in {topic}")
    print("=" * 60)
    
    analysis = await inspector.analyze_dlq_patterns(topic, limit=limit)
    
    if not analysis:
        print("No analysis data available.")
        return
    
    print(f"Total Messages Analyzed: {analysis.get('total_messages', 0)}")
    
    # Error patterns
    error_patterns = analysis.get('error_patterns', {})
    if error_patterns:
        print("\n🚨 Error Patterns:")
        for error_type, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True):
            print(f"   {error_type}: {count}")
    
    # Service distribution
    service_dist = analysis.get('service_distribution', {})
    if service_dist:
        print("\n🏢 Service Distribution:")
        for service, count in sorted(service_dist.items(), key=lambda x: x[1], reverse=True):
            print(f"   {service}: {count}")
    
    # Retry distribution
    retry_dist = analysis.get('retry_distribution', {})
    if retry_dist:
        print("\n🔄 Retry Distribution:")
        for retry_count, count in sorted(retry_dist.items()):
            print(f"   {retry_count} retries: {count}")

async def export_messages(inspector: DLQInspector, topic: str, format: str, limit: int, output_file: str = None):
    """Export DLQ messages."""
    print(f"\n📤 Exporting messages from {topic}")
    print("=" * 60)
    
    export_data = await inspector.export_dlq_messages(topic, format=format, limit=limit)
    
    if not export_data:
        print("No data to export.")
        return
    
    if output_file:
        # Write to file
        with open(output_file, 'w') as f:
            f.write(export_data)
        print(f"✅ Data exported to {output_file}")
    else:
        # Print to console
        print(export_data)

def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="DLQ Management Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List messages in a DLQ topic
  python dlq_tools.py list usdcop-data.dlq --limit 10
  
  # Show summary of all DLQ topics
  python dlq_tools.py summary
  
  # Replay a specific message
  python dlq_tools.py replay usdcop-data.dlq msg_123
  
  # Purge old messages
  python dlq_tools.py purge usdcop-data.dlq --days 7
  
  # Analyze patterns
  python dlq_tools.py analyze usdcop-data.dlq --limit 100
  
  # Export messages
  python dlq_tools.py export usdcop-data.dlq --format json --output dlq_export.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List messages in DLQ topic')
    list_parser.add_argument('topic', help='DLQ topic name')
    list_parser.add_argument('--limit', type=int, default=50, help='Maximum number of messages to show')
    list_parser.add_argument('--offset', type=int, default=0, help='Offset for pagination')
    
    # Summary command
    subparsers.add_parser('summary', help='Show summary of all DLQ topics')
    
    # Replay command
    replay_parser = subparsers.add_parser('replay', help='Replay a message from DLQ')
    replay_parser.add_argument('topic', help='DLQ topic name')
    replay_parser.add_argument('message_id', help='Message ID to replay')
    replay_parser.add_argument('--target', help='Target topic for replay (defaults to original topic)')
    
    # Purge command
    purge_parser = subparsers.add_parser('purge', help='Purge old messages from DLQ')
    purge_parser.add_argument('topic', help='DLQ topic name')
    purge_parser.add_argument('--days', type=int, default=7, help='Remove messages older than N days')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze patterns in DLQ messages')
    analyze_parser.add_argument('topic', help='DLQ topic name')
    analyze_parser.add_argument('--limit', type=int, default=1000, help='Maximum number of messages to analyze')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export DLQ messages')
    export_parser.add_argument('topic', help='DLQ topic name')
    export_parser.add_argument('--format', choices=['json', 'csv'], default='json', help='Export format')
    export_parser.add_argument('--limit', type=int, default=1000, help='Maximum number of messages to export')
    export_parser.add_argument('--output', help='Output file path (defaults to console)')
    
    return parser

async def main():
    """Main function."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Initialize DLQ inspector (without actual connections for CLI)
        inspector = DLQInspector()
        
        if args.command == 'list':
            await list_dlq_messages(inspector, args.topic, args.limit, args.offset)
        
        elif args.command == 'summary':
            await show_dlq_summary(inspector)
        
        elif args.command == 'replay':
            await replay_message(inspector, args.topic, args.message_id, args.target)
        
        elif args.command == 'purge':
            await purge_old_messages(inspector, args.topic, args.days)
        
        elif args.command == 'analyze':
            await analyze_patterns(inspector, args.topic, args.limit)
        
        elif args.command == 'export':
            await export_messages(inspector, args.topic, args.format, args.limit, args.output)
        
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
